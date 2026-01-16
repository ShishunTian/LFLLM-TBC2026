import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, DEFAULT_DEPTH_TOKEN, DEFAULT_DEP_START_TOKEN, DEFAULT_DEP_END_TOKEN, 
                    DEFAULT_AROUND_TOKEN, DEFAULT_AR_START_TOKEN, DEFAULT_AR_END_TOKEN)
from .vqa_dataset import VQADataset
import sys
sys.path.append('/home/xxx/project/LFLLM/model/llava/model')
from model.warp_to_center import WarpFusion

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    images_left_list = []
    images_right_list = []
    images_top_list = []
    images_bottom_list = []
    images_depth_list = []
    depth_images_clip_list = []
    # around_images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        depth_images_clip,
        images_left,
        images_right,
        images_top,
        images_bottom,
        images_depth,
        # around_images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        depth_images_clip_list.append(depth_images_clip)
        images_left_list.append(images_left)
        images_right_list.append(images_right)
        images_top_list.append(images_top)
        images_bottom_list.append(images_bottom)
        images_depth_list.append(images_depth)
        # around_images_clip_list.append(around_images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
            replace_token = DEFAULT_DEPTH_TOKEN
            replace_token = (
                DEFAULT_DEP_START_TOKEN + replace_token + DEFAULT_DEP_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_DEPTH_TOKEN, replace_token
            )
            # replace_token = DEFAULT_AROUND_TOKEN
            # replace_token = (
            #     DEFAULT_AR_START_TOKEN + replace_token + DEFAULT_AR_END_TOKEN
            # )
            # conversation_list[i] = conversation_list[i].replace(
            #     DEFAULT_AROUND_TOKEN, replace_token
            # )
            
    # print("this is the second step")
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    # if inferences[0] == True:
    #     for prompt in conversation_list:
    #         print("depth_token_indices",prompt)
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # if inferences[0] == True:
    #     print("depth_token_indices",input_ids[0])

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation: # TODO
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    # print("inferences[0]",inferences[0])
    
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            print("input_ids",input_ids.shape)
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
    
    #     depth_token_indices = torch.where(input_ids == -200)[0]
    #     print("depth_token_indices",depth_token_indices)
    
    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "depth_images_clip": torch.stack(depth_images_clip_list, dim=0),
        "images_left": torch.stack(images_left_list, dim=0),
        "images_right": torch.stack(images_right_list, dim=0),
        "images_top": torch.stack(images_top_list, dim=0),
        "images_bottom": torch.stack(images_bottom_list, dim=0),
        "images_depth": torch.stack(images_depth_list, dim=0),
        # "around_images_clip": torch.stack(around_images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        self.warp_fusion = WarpFusion()
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)# 这里的split 可以自己选定 val  o-1
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                elif ds in ["ref-urbanlf-syn"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/urbanlf/syn/val/JPEGImages",
                        item["file_name"],
                    )
                elif ds in ["ref-urbanlf-real"]:
                    item["file_name"] = os.path.join(
                        "/mnt/mdisk/xxx/LFLLM/dataset/rlfos/Real-val/JPEGImages",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_path_origin = image_path
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])
            sampled_sents = sents
            # print("sampled_sents=======     ",sampled_sents)
            sampled_ann_ids = ann_ids
            # print("image_path=======     ",image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 这一段视差代码一定在测试一张图片的时候要注释，影响到测试性能
            # depth_path = "/data1/xxx/LFLLM/dataset/animal/oavc_npfile/image_" + str(f"{image_id:05d}") + "__fullres.npy"
            # depth_path = "/data1/xxx/LFLLM/dataset/dataset-test/np_file_distill/image_" + str(f"{image_id:05d}") + "_.npy"
            depth_path = "/data1/xxx/LFLLM/dataset/syn/Image" + str(image_id) + "/5_5_disparity_OAVC.npy"
            # depth_path = "/mnt/mdisk/xxx/BarLeRIa/datasets/raw/real/Image" + str(image_id) + "/disparity_OAVC.npy"
            depth_arr = np.load(depth_path)
            # 原始范围的最小值和最大值
            min_original = -1
            max_original = 1
            # 将超出范围的值限制在0和1之间
            depth_arr = np.clip((depth_arr - min_original) / (max_original - min_original), 0, 1)
            depth_arr = (depth_arr * 255).astype('uint8')
            depth_image = np.stack((depth_arr,) * 3, axis=-1)
            # depth_image = cv2.imread(depth_path)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

            # 初始化一个空列表来存储筛选出的图片数组
            images_arrays = []
            # 遍历文件夹中的所有文件
            # folder_path = "/data1/xxx/LFLLM/dataset/animal/full/image_" + str(f"{image_id:05d}") + "__fullres"
            folder_path = "/data1/xxx/LFLLM/dataset/syn/Image" + str(image_id)
            for filename in os.listdir(folder_path):
                if filename.endswith('5_4.png'):
                    # print(filename)
                    image_path = os.path.join(folder_path, filename)
                    image_left = cv2.imread(image_path)
                    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
                    image_left = self.clip_image_processor.preprocess(image_left, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    # 为SAM做准备
                    # image_around = self.transform.apply_image(image_around)
                    # resize = image_around.shape[:2]
                    # image_around = self.preprocess(torch.from_numpy(image_around).permute(2, 0, 1).contiguous())

                    # image_around = torch.as_tensor(image_around, dtype=torch.bfloat16)
                    # images_arrays.append(image_around)
                if filename.endswith('5_6.png'):
                    # print(filename)
                    image_path = os.path.join(folder_path, filename)
                    image_right = cv2.imread(image_path)
                    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
                    image_right = self.clip_image_processor.preprocess(image_right, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                if filename.endswith('4_5.png'):
                    # print(filename)
                    image_path = os.path.join(folder_path, filename)
                    image_top = cv2.imread(image_path)
                    image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB)
                    image_top = self.clip_image_processor.preprocess(image_top, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                if filename.endswith('6_5.png'):
                    # print(filename)
                    image_path = os.path.join(folder_path, filename)
                    image_bottom = cv2.imread(image_path)
                    image_bottom = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2RGB)
                    image_bottom = self.clip_image_processor.preprocess(image_bottom, return_tensors="pt")[
                        "pixel_values"
                    ][0]

            is_sentence = False
        else:
            image_path = self.images[idx]
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            text = text.rstrip(".")
            # if is_sentence:
            #     conv.append_message(
            #         conv.roles[0],
            #         DEFAULT_IMAGE_TOKEN
            #         + "\n {} Please output segmentation mask.".format(text),
            #     )
            #     conv.append_message(conv.roles[1], "[SEG].")
            # else:
            #     conv.append_message(
            #         conv.roles[0],
            #         DEFAULT_IMAGE_TOKEN
            #         + "\n What is {} in this image? Please output segmentation mask.".format(
            #             text
            #         ),
            #     )
            #     conv.append_message(conv.roles[1], "[SEG].")
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    # DEFAULT_AROUND_TOKEN +
                    DEFAULT_DEPTH_TOKEN +
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    # DEFAULT_AROUND_TOKEN +
                    DEFAULT_DEPTH_TOKEN +
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # image_depth_clip = images_arrays[0]
        # image_left = images_arrays[0]
        # image_right = images_arrays[1]
        # image_around_clip = images_arrays[1]
        image_depth_clip = self.clip_image_processor.preprocess(depth_image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_depth = torch.as_tensor(depth_image, dtype=torch.bfloat16)

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True
        
        return (
            image_path_origin,
            image,
            image_clip,
            image_depth_clip,
            image_left,
            image_right,
            image_top,
            image_bottom,
            image_depth,
            # image_around_clip,
            conversations,
            masks,
            labels,
            resize,
            sampled_sents,
            None,
            inference,
        )
