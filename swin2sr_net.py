# -*- coding: utf-8 -*-
"""
@author: Sreeraman SR
"""

import argparse
import base64
import concurrent.futures
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import requests
import torch

from models.network_swin2sr import Swin2SR as net
from utils import util_calculate_psnr_ssim as util


class ImageResolutionUpScaler:
    task = "real_sr"
    scale = 4  # 4
    window_size = 8  # 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_path, image_paths):
        self.model_path = model_path
        self.image_paths = image_paths
        self.model = self.load_model()
        self.model.eval()
        self.model = self.model.to(self.device)

    def load_model(self):
        # real-world image sr
        if self.task == "real_sr":
            # used 'nearest+conv' to avoid block artifacts default embed_dim = 180, num_head= 6,depth=6
            model = net(
                upscale=self.scale,
                in_chans=3,
                img_size=64,
                window_size=self.window_size,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv",
            )

        param_key_g = "params_ema"
        pretrained_model = torch.load(self.model_path)
        model.load_state_dict(
            pretrained_model[param_key_g]
            if param_key_g in pretrained_model.keys()
            else pretrained_model,
            strict=True,
        )

        return model

    def imageSuperResolution(self, image_path):
        # Decode the image string and convert it to a numpy array incase of base64 data
        # img_data = base64.b64decode(image_path.split(",")[1])
        # np_arr = np.frombuffer(img_data, np.uint8)  # Corrected line

        # # Decode the numpy array into an OpenCV image
        # img_lq = cv2.imdecode(np_arr, cv2.IMREAD_COLOR).astype(np.float32) / 255

        # Read the image file
        img_lq = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
        )  # HCW-BGR to CHW-RGB
        img_lq = (
            torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)
        )  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                :, :, : h_old + h_pad, :
            ]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                :, :, :, : w_old + w_pad
            ]
            output = self.model(img_lq)

            output = output[..., : h_old * self.scale, : w_old * self.scale]

        # output image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        return output

    # Processing multiple images concurrently
    def process_images_parallel(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [
                executor.submit(self.imageSuperResolution, path)
                for path in self.image_paths
            ]

            concurrent.futures.wait(results)
            return results
