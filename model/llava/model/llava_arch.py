#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                         IMAGE_TOKEN_INDEX,
                         DEPTH_TOKEN_INDEX,DEFAULT_DEPTH_TOKEN,DEFAULT_DEP_START_TOKEN,DEFAULT_DEP_END_TOKEN,AROUND_TOKEN_INDEX)

from .multimodal_encoder.builder import build_vision_tower


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
        , depth_images
        # , around_images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            # print("depth_images.shape",depth_images.shape)
            # depth_concat_images = torch.cat([image for image in depth_images], dim=0)
            # print("depth_concat_images.shape",depth_concat_images.shape)
            image_features = self.encode_images(concat_images)
            # depth_image_features = self.encode_images(depth_concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # depth_image_features = torch.split(depth_image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features] #  x 是一个形状为 (batch_size, num_features) 的张量，x.flatten(0, 1) 会将这个张量转换成形状为 (batch_size * num_features) 的一维张量
            # depth_image_features = [x.flatten(0, 1) for x in depth_image_features]
        else:
            image_features = self.encode_images(images)
            # B1, C1, H1, W1= depth_images.shape
            # decomposed_images = depth_images.view(B1, 2, 3, 224, 224)
            # decomposed_images = torch.unbind(decomposed_images, dim=1)
            depth_image_features = self.encode_images(depth_images)
            # print("image_around_clip    2",depth_images.dtype)
            # around_image_features = self.encode_images(around_images)
            # de_images = []
            # for de_image in decomposed_images:
            #     de_image_features = self.encode_images(de_image)
            #     de_images.append(de_image_features)
            # de_images = torch.cat(de_images, dim=1)
            # # print("depth shape",de_images.shape)
            # de_images = de_images.unsqueeze(3)
            # self.one_by_one_conv1 = OneByOneConvLayer(in_channels=256 * 4, out_channels=256).cuda().to(torch.bfloat16)
            # depth_image_features = self.one_by_one_conv1(de_images)
            # depth_image_features = depth_image_features.squeeze(3)

            # de_images = self.one_by_one_conv(de_images)
            # print("depth shape",depth_images.shape)  # torch.Size([8, 3, 224, 224])
            # depth_image_features = self.encode_images(depth_images)
            # print("depth_image_features",depth_image_features.shape) # torch.Size([8, 256, 4096])

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # print("this is the fifth step3",input_ids)
        # depth_token_indices = torch.where(input_ids == -200)[0]
        # print("depth_token_indices",depth_token_indices)
        for batch_idx, cur_input_ids in enumerate(input_ids): # # 遍历 input_ids 张量的每一行 索引是batch_idx  每一行值为  cur_input_ids
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: # # 如果序列中没有任何元素等于 IMAGE_TOKEN_INDEX，则条件为真。
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            depth_token_indices = torch.where(cur_input_ids == DEPTH_TOKEN_INDEX)[0]
            # around_token_indices = torch.where(cur_input_ids == AROUND_TOKEN_INDEX)[0]
            # print("image_token_indices",image_token_indices)
            # print("depth_token_indices",cur_input_ids)
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                cur_depth_image_features = depth_image_features[cur_image_idx]
                # cur_around_image_features = around_image_features[cur_image_idx]

                image_token_start = image_token_indices[0]
                depth_token_start = depth_token_indices[0]
                # around_token_start = around_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False # getattr 函数用于获取对象（在这里是 self.config）的属性。如果属性存在，则返回属性值，如果不存在，则返回默认值（在这里是 False）
                ):
                    # cur_new_input_embeds.append(
                    #     self.get_model()
                    #     .embed_tokens(cur_input_ids[: around_token_start - 1])
                    #     .detach()
                    # )
                    # cur_new_input_embeds.append(
                    #     self.get_model()
                    #     .embed_tokens(cur_input_ids[around_token_start - 1 : around_token_start])
                    # )
                    # cur_new_input_embeds.append(cur_around_image_features)
                    # cur_new_input_embeds.append(
                    #     self.get_model()
                    #     .embed_tokens(cur_input_ids[around_token_start + 1: around_token_start - 1])
                    # )
                    # cur_new_input_embeds.append(
                    #     self.get_model()
                    #     .embed_tokens(cur_input_ids[around_token_start - 1: depth_token_start - 1])
                    #     .detach()
                    # )
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: depth_token_start - 1])
                    )
                    cur_new_input_embeds.append(cur_depth_image_features)
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[depth_token_start + 1: image_token_start - 1])
                    )
                    # cur_new_input_embeds.append(
                    #     self.get_model()
                    #     .embed_tokens(cur_input_ids[: image_token_start - 1])
                    #     .detach()
                    # ) # 修改的 .detach(): 在第一个嵌入操作中使用 .detach()，这可能是为了确保这部分嵌入不会被后续的梯度更新所影响，即它们不会参与梯度回传。
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        # cur_new_labels.append(cur_labels[:around_token_start])
                        # cur_new_labels.append(
                        #     torch.full(
                        #         (cur_around_image_features.shape[0],),
                        #         IGNORE_INDEX,
                        #         device=labels.device,
                        #         dtype=labels.dtype,
                        #     )
                        # )

                        cur_new_labels.append(cur_labels[:depth_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_depth_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )

                        cur_new_labels.append(cur_labels[depth_token_start:image_token_start])
                        # cur_new_labels.append(cur_labels[:image_token_start]) # 修改的
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                        
                elif getattr(self.config, "mm_use_im_start_end", False):
                    # cur_new_input_embeds.append(
                    #     self.get_model().embed_tokens(cur_input_ids[:around_token_start])
                    # )
                    # cur_new_input_embeds.append(cur_around_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[: depth_token_start])
                    )
                    cur_new_input_embeds.append(cur_depth_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[depth_token_start + 1 : image_token_start])
                    )
                    # cur_new_input_embeds.append(
                    #     self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    # ) # 修改的
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        # cur_new_labels.append(cur_labels[:around_token_start])
                        # cur_new_labels.append(
                        #     torch.full(
                        #         (cur_around_image_features.shape[0],),
                        #         IGNORE_INDEX,
                        #         device=labels.device,
                        #         dtype=labels.dtype,
                        #     )
                        # )
                        cur_new_labels.append(cur_labels[: depth_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_depth_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(cur_labels[depth_token_start + 1 : image_token_start])
                        # cur_new_labels.append(cur_labels[:image_token_start]) # 修改的
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    # print("this is the fifth step2")
                    # cur_new_input_embeds.append(
                    #     self.get_model().embed_tokens(cur_input_ids[:around_token_start])
                    # )
                    # cur_new_input_embeds.append(cur_around_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[: depth_token_start])
                    )
                    cur_new_input_embeds.append(cur_depth_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[depth_token_start + 1 : image_token_start])
                    )
                    # cur_new_input_embeds.append(
                    #     self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    # ) # 修改的
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        # cur_new_labels.append(cur_labels[:around_token_start])
                        # cur_new_labels.append(
                        #     torch.full(
                        #         (cur_around_image_features.shape[0],),
                        #         IGNORE_INDEX,
                        #         device=labels.device,
                        #         dtype=labels.dtype,
                        #     )
                        # )
                        cur_new_labels.append(cur_labels[: depth_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_depth_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(cur_labels[depth_token_start + 1 : image_token_start])
                        # cur_new_labels.append(cur_labels[:image_token_start])# 修改的
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # self.resize_token_embeddings(len(tokenizer))

            # if num_new_tokens > 0:
            #     input_embeddings = self.get_input_embeddings().weight.data
            #     output_embeddings = self.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
