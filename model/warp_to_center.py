import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import cv2

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class WarpFusion():

        
    def warp(self, disp, left, right, top, bottom, center):
        # print("center.shape",center.shape) # center.shape torch.Size([2, 256, 64, 64])
        # print("left.shape",left.shape) # left.shape torch.Size([2, 256, 64, 64])
        # print("disp.shape",disp.shape)
        disp_add = disp
        disp = disp.permute(0,3,1,2)
        disp = resize(
                disp,
                size=(224,224),
                mode='nearest',
                warning=False)
        disp = disp.permute(0,2,3,1) # (B, C, H, W)  ->  (B, H, W, C)   
        B, height, width, _ = disp.shape # (B, H, W, C)  
        disp = disp[:, :, :, 0]
        disp = disp.unsqueeze(3)
        
        x = np.array([i for i in range(0, height)]).reshape(1, height, 1, 1).repeat(repeats=width,axis=2)  # x 和 y 坐标网格，这些坐标代表图像中每个像素的位置
        y = np.array([i for i in range(0, width)]).reshape(1, 1, width, 1).repeat(repeats=height, axis=1)
        xy_position = torch.from_numpy(np.concatenate([x, y], axis=-1))  # x 和 y 坐标合并为一个张量，表示每个像素的位置

        coords_x = torch.linspace(-1, 1, width).to(torch.bfloat16)  # 标准化的坐标 coords_x 和 coords_y，它们的范围在 -1 到 1 之间
        coords_y = torch.linspace(-1, 1, height).to(torch.bfloat16)
        coords_x = coords_x.repeat(height, 1).reshape(height, width, 1)
        coords_y = coords_y.repeat(width, 1).permute(1, 0).reshape(height, width, 1)
        coords = torch.cat([coords_x, coords_y], dim=2)
        coords = coords.reshape(1, height, width, 2)
        coords = coords[0, xy_position[:, :, :, 0].reshape(-1).to(torch.int64),
                xy_position[:, :, :, 1].reshape(-1).to(torch.int64), :]  # 基于裁剪的patchsize像素块选出对应的coords值
        coords = coords.reshape(-1, height, width, 2).cuda()    
        coords = coords.repeat(B, 1, 1, 1)
        # print("coords.shape",coords.shape)
        warp_features = []

        dst_u = 5
        dst_v = 5
        # sai_number, batch_size, channels, height, width = sequence_imgs.shape
        sequence_index = [[5,6],[5,4],[6,5],[4,5]]  
        sequence_index = torch.tensor(sequence_index).unsqueeze(0)
        sequence_index = sequence_index.repeat(B, 1, 1)
        offsetx_right = 2 * (dst_u - sequence_index[:,0,1]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]
        offsety_right = 2 * (dst_v - sequence_index[:,0,0]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]     #  计算垂直方向上的偏移量。这里 dst_v 是目标v坐标，sequence_index[:,i,0] 是当前SAI的v坐标，disparity_8 是视差图。
        # print("offsety_right.shape",offsety_right.shape)
        coords_x_right = (coords[:, :, :, 0:1] * width + offsetx_right) / width
        coords_y_right = (coords[:, :, :, 1:2] * height + offsety_right) / height
        coords_uv_right = torch.cat([coords_x_right, coords_y_right], dim=-1).to(torch.bfloat16)
        # print("coords_uv_right.shape",coords_uv_right.shape) # shape torch.Size([2, 480, 640, 2])
        offsetx_left = 2 * (dst_u - sequence_index[:,1,1]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]
        offsety_left = 2 * (dst_v - sequence_index[:,1,0]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]     #  计算垂直方向上的偏移量。这里 dst_v 是目标v坐标，sequence_index[:,i,0] 是当前SAI的v坐标，disparity_8 是视差图。
        coords_x_left = (coords[:, :, :, 0:1] * width + offsetx_left) / width
        coords_y_left = (coords[:, :, :, 1:2] * height + offsety_left) / height
        # print("coords_x_left.shape",coords_x_left.shape)
        coords_uv_left = torch.cat([coords_x_left, coords_y_left], dim=-1).to(torch.bfloat16)

        offsetx_bottom = 2 * (dst_u - sequence_index[:,2,1]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]
        offsety_bottom = 2 * (dst_v - sequence_index[:,2,0]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]     #  计算垂直方向上的偏移量。这里 dst_v 是目标v坐标，sequence_index[:,i,0] 是当前SAI的v坐标，disparity_8 是视差图。
        coords_x_bottom = (coords[:, :, :, 0:1] * width + offsetx_bottom) / width
        coords_y_bottom = (coords[:, :, :, 1:2] * height + offsety_bottom) / height
        # print("coords_x_left.shape",coords_x_left.shape)
        coords_uv_bottom = torch.cat([coords_x_bottom, coords_y_bottom], dim=-1).to(torch.bfloat16)

        offsetx_top = 2 * (dst_u - sequence_index[:,3,1]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]
        offsety_top = 2 * (dst_v - sequence_index[:,3,0]).reshape(B,1,1,1).cuda() * disp[:, :, :, :]     #  计算垂直方向上的偏移量。这里 dst_v 是目标v坐标，sequence_index[:,i,0] 是当前SAI的v坐标，disparity_8 是视差图。
        coords_x_top = (coords[:, :, :, 0:1] * width + offsetx_top) / width
        coords_y_top = (coords[:, :, :, 1:2] * height + offsety_top) / height
        # print("coords_x_left.shape",coords_x_left.shape)
        coords_uv_top = torch.cat([coords_x_top, coords_y_top], dim=-1).to(torch.bfloat16)

        res_right = F.grid_sample(right, coords_uv_right[:, :, :, :],mode='bilinear',padding_mode='border') # 使用双线性插值对当前特征进行采样，根据计算出的坐标。

        res_left = F.grid_sample(left, coords_uv_left[:, :, :, :],mode='bilinear',padding_mode='border') # 使用双线性插值对当前特征进行采样，根据计算出的坐标。

        res_bottom = F.grid_sample(bottom, coords_uv_bottom[:, :, :, :],mode='bilinear',padding_mode='border') # 使用双线性插值对当前特征进行采样，根据计算出的坐标。

        res_top = F.grid_sample(top, coords_uv_top[:, :, :, :],mode='bilinear',padding_mode='border') # 使用双线性插值对当前特征进行采样，根据计算出的坐标。
        # print("res_top.shape",res_top.shape) # torch.Size([2, 256, 480, 640])
        # print("res_left.shape",res_left.shape)
        # print("res_right.shape",res_right.shape)
        # print("res_bottom.shape",res_bottom.shape)
        # print("center.shape",center.shape)
        feature_maps = []
        feature_maps.append(res_left)
        feature_maps.append(res_right)
        feature_maps.append(res_top)
        feature_maps.append(res_bottom)
        feature_maps.append(center)
        weights = [0.1, 0.1, 0.1, 0.1, 0.6]  # 定义每个特征图的权重

        # 计算加权平均
        weighted_sum = sum(w * f for w, f in zip(weights, feature_maps))
        averaged_feature_map = weighted_sum / sum(weights)
        return averaged_feature_map