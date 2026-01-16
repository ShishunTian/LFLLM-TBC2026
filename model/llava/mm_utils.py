import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import StoppingCriteria
import numpy as np

from .constants import IMAGE_TOKEN_INDEX, DEFAULT_DEPTH_TOKEN,DEPTH_TOKEN_INDEX,AROUND_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors="pt")["pixel_values"]


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, depth_token_index=DEPTH_TOKEN_INDEX, around_token_index=AROUND_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    depth_prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<depth>")]
    around_prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<around>")]

    # 定义标签
    # labels = ( "<around>", "<depth>", "<image>" )
    # labels = ("<depth>", "<image>" )
    # # 按照标签分割提示字符串
    # new_arr = []
    # chunks = []
    # temp_chunk = ""
    # for label in labels:
    #     temp_chunk, prompt = prompt.split(label, 1)
    #     new_arr = new_arr + tokenizer(temp_chunk).input_ids
    #     if label == "<around>" :
    #         new_arr.append(-300)
    #     elif label == "<depth>" :
    #         new_arr.append(-400)
    #     elif label == "<image>" :
    #         new_arr.append(-200)
    #         new_arr = new_arr + tokenizer(prompt).input_ids

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 这个在测试一张图片使用代码和2.结合
    # for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
    #     #print("before datasets inputids",x)
    #     input_ids.extend(x[offset:])
    
    # 插入深度信息标记
    depth_inserted_chunks = insert_separator(depth_prompt_chunks, [depth_token_index])
    for chunk in depth_inserted_chunks:
        input_ids.extend(chunk)
    # print("before datasets inputids",input_ids)
    # 遍历数组并进行处理
    new_arr = []
    replace = False
    for num in input_ids:
        if num == 32001:
            replace = True
            new_arr.append(32001)
            new_arr.append(-200)  # 在32001的位置添加-200
        elif num == 32002:
            replace = False
            if not replace:
                new_arr.append(num)  # 在32002的位置添加该数字
        elif not replace:
            new_arr.append(num)  # 只有在不处于32001和32002之间时才添加数字
    new_arr.pop(0)
    # print("after datasets inputids",new_arr)
    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(new_arr, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return new_arr
    # 2.这是第二段代码和1.结合
    # if return_tensors is not None:
    #     if return_tensors == "pt":
    #         return torch.tensor(input_ids, dtype=torch.long)
    #     raise ValueError(f"Unsupported tensor type: {return_tensors}")
    # return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
