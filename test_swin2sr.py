# -*- coding: utf-8 -*-
"""
@author: Sreeraman SR
"""

import os

import cv2

from swin2sr_net import ImageResolutionUpScaler

# Modal path
model_pth = "./models/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"

# Save images
save_dir = "./results"

# path to images
image_dir = "./inputs"

# Get all files in the directory
files = os.listdir(image_dir)

# Getting the image files along with path
image_paths = [os.path.join(image_dir, file) for file in files if file.endswith(".jpg")]

upscaler = ImageResolutionUpScaler(model_pth, image_paths)  # Model path, image path

# new_res = upscaler.imageSuperResolution(image_pth + "/" + "img1.jpg")  # image path
output_images = upscaler.process_images_parallel()

# Show image
for output in output_images:
    cv2.imshow("Upscaled image", output.result())
    cv2.waitKey()
    cv2.destroyAllWindows()

    # # Save Image
    # cv2.imwrite(save_dir + "/" + "Swin2RealSR2.png", output)
