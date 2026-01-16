import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LFLLM import LFLLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, DEFAULT_DEPTH_TOKEN, DEFAULT_DEP_START_TOKEN, DEFAULT_DEP_END_TOKEN, 
                    DEFAULT_AROUND_TOKEN, DEFAULT_AR_START_TOKEN, DEFAULT_AR_END_TOKEN)
from model.warp_to_center import WarpFusion


def parse_args(args):
    parser = argparse.ArgumentParser(description="LFLLM chat")
    parser.add_argument("--version", default="checkpoint/LFLLM-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LFLLMForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_DEPTH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            replace_token = (
                DEFAULT_DEP_START_TOKEN + DEFAULT_DEPTH_TOKEN + DEFAULT_DEP_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_DEPTH_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        data_type = "real"
        image_id = input("Please input the image Id: ")
        image_path = "/mnt/mdisk/xxx/BarLeRIa/datasets/raw/" + data_type + "/Image" + str(image_id) + "/5_5.png"
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        depth_path = "/mnt/mdisk/xxx/BarLeRIa/datasets/raw/" + data_type + "/Image" + str(image_id) + "/disparity_OAVC.npy"
        # depth_path = "/mnt/mdisk/xxx/BarLeRIa/datasets/raw/real/Image" + str(image_id) + "/disparity_OAVC.npy"
        depth_arr = np.load(depth_path)
        # 原始范围的最小值和最大值
        min_original = -0.5
        max_original = 1.5
        # 将超出范围的值限制在0和1之间
        depth_arr = np.clip((depth_arr - min_original) / (max_original - min_original), 0, 1)
        depth_arr = (depth_arr * 255).astype('uint8')
        depth_image = np.stack((depth_arr,) * 3, axis=-1)
        # depth_image = cv2.imread(depth_path)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        folder_path = "/mnt/mdisk/xxx/BarLeRIa/datasets/raw/" + data_type + "/Image" + str(image_id)
        for filename in os.listdir(folder_path):
            if filename.endswith('5_4.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_left = cv2.imread(image_path)
                image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
                image_left = (
                    clip_image_processor.preprocess(image_left, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    .unsqueeze(0)
                    .cuda()
                )
            if filename.endswith('5_6.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_right = cv2.imread(image_path)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
                image_right = (
                    clip_image_processor.preprocess(image_right, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    .unsqueeze(0)
                    .cuda()
                )
            if filename.endswith('4_5.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_top = cv2.imread(image_path)
                image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB)
                image_top = (
                    clip_image_processor.preprocess(image_top, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    .unsqueeze(0)
                    .cuda()
                )
            if filename.endswith('6_5.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_bottom = cv2.imread(image_path)
                image_bottom = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2RGB)
                image_bottom = (
                    clip_image_processor.preprocess(image_bottom, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    .unsqueeze(0)
                    .cuda()
                )

        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        image_depth_clip = (clip_image_processor.preprocess(depth_image, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
        )
        image_depth = torch.as_tensor(depth_image, dtype=torch.bfloat16)

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
            image_depth_clip = image_depth_clip.bfloat16()
            image_left = image_left.bfloat16()
            image_right = image_right.bfloat16()
            image_top = image_top.bfloat16()
            image_bottom = image_bottom.bfloat16()
            image_depth = image_depth.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        warp_fusion = WarpFusion()

        image_clip = warp_fusion.warp(image_depth_clip, image_left, image_right, image_top, image_bottom, image_clip)

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image_depth_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX] # 筛选出所有不等于 IMAGE_TOKEN_INDEX 的元素

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
