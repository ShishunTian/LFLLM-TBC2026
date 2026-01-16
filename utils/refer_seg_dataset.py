import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
import sys
sys.path.append('/home/xxx/project/LFLLM/model/llava/model')
from model.warp_to_center import WarpFusion

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
    ):
        self.warp_fusion = WarpFusion()

        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")  #选定训练数据集
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                #print(ds)
                #input('.l77 refer_seg_dataset.py.')
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds == "ref-urbanlf-syn":
                    item["file_name"] = os.path.join(
                        "/mnt/mdisk/xxx/LFLLM/dataset/rlfos/Syn-train/JPEGImages", item["file_name"]
                    )
                elif ds == "ref-urbanlf-real":
                    item["file_name"] = os.path.join(
                        "/mnt/mdisk/xxx/LFLLM/dataset/rlfos/Real-train/JPEGImages", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

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
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                text = text.rstrip(".")
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        # print("sents=======     ",sents)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # print("sampled_sents=======     ",sampled_sents, self.num_classes_per_sample)
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_path = "/data1/xxx/LFLLM/dataset/animal/oavc_npfile/image_" + str(f"{image_id:05d}") + "__fullres.npy"
        # depth_path = "/data1/xxx/LFLLM/dataset/dataset-test/np_file_distill/image_" + str(f"{image_id:05d}") + "_.npy"
        # depth_path = "/mnt/mdisk/xxx/BarLeRIa/datasets/raw/syn/Image" + str(image_id) + "/5_5_disparity_OAVC.npy"
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
        folder_path = "/data1/xxx/LFLLM/dataset/animal/full/image_" + str(f"{image_id:05d}") + "__fullres"
        for filename in os.listdir(folder_path):
            if filename.endswith('view_2_1.png'):
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
            if filename.endswith('view_2_3.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_right = cv2.imread(image_path)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
                image_right = self.clip_image_processor.preprocess(image_right, return_tensors="pt")[
                    "pixel_values"
                ][0]
            if filename.endswith('view_1_2.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_top = cv2.imread(image_path)
                image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB)
                image_top = self.clip_image_processor.preprocess(image_top, return_tensors="pt")[
                    "pixel_values"
                ][0]
            if filename.endswith('view_3_2.png'):
                # print(filename)
                image_path = os.path.join(folder_path, filename)
                image_bottom = cv2.imread(image_path)
                image_bottom = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2RGB)
                image_bottom = self.clip_image_processor.preprocess(image_bottom, return_tensors="pt")[
                    "pixel_values"
                ][0]

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image_depth_clip = self.clip_image_processor.preprocess(depth_image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_depth = torch.as_tensor(depth_image, dtype=torch.bfloat16)
        # print("image_depth_clip.shape111", image_depth_clip.shape)
        # image_depth_clip = torch.cat(images_arrays, dim=0)
        # image_left = images_arrays[0]
        # image_right = images_arrays[1]

        # image_around_clip = images_arrays[1]
        # print("image_depth_clip    1",image_depth_clip.dtype)
        # print("image_around_clip    1",image_around_clip.dtype)

        # print("image_depth_clip.shape222", image_depth_clip.shape)
        # print("this is the first step")
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
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
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
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

        masks = np.stack(masks, axis=0)

        # if ds == 'grefcoco' and flag:
        #     import shutil
        #     image_name = image_path.split("/")[-1]
        #     save_dir = os.path.join("/group/30042/xlai/LFLLM_refactor_final/debug", image_name.split(".")[0])
        #     os.makedirs(save_dir, exist_ok=True)
        #     shutil.copy(image_path, save_dir)
        #     for i in range(masks.shape[0]):
        #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
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
            label,
            resize,
            questions,
            sampled_classes,
        )
