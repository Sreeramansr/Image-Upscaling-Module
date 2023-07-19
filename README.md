"""

## [Image Upscaling Module]

""" 
This module provides functionality for upscaling low quality, low resolution images using a thread pool for concurrent processing. It utilizes an object-oriented approach and is implemented in Python.
The code is based on the following research paper 
## [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345).
I have used the pre-trained model weights provided in following git link ((https://github.com/mv-lab/swin2sr/releases)
Official repository)


# Features

Image upscaling using a deep learning model
Concurrent processing of multiple images using a thread pool
Thread-safe execution

# Requirements

Python >= 3.8
torch (PyTorch library)
opencv-python (OpenCV library)
numpy

# Installation

Install the required Python libraries: 
“ pip install -r requirements.txt”

# Usage

1.	Import the ImageResolutionUpScaler class from the module:

    “””
    “from image_upscaler import ImageResolutionUpScaler “

    “””

2.	Create an instance of the ImageResolutionUpScaler class by providing the model path and image paths:

    “””
    model_path = "path/to/model.pth" 
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    upscaler = ImageResolutionUpScaler(model_path, image_paths) 

    “””

3.	Process the images concurrently using a thread pool:

    “””
    results = upscaler.process_images_parallel() 

    “””

4.	Access the results of the image processing tasks:

    “”
    for result in results:
        output_image = result.result()

    ""

    # Display or save the images
        cv2.imshow("Upscaled image", output_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    “””

# Example

An example script test_swin2sr.py is provided to demonstrate the usage of the ImageResolutionUpScaler module.

# Acknowledgments

The codes are based on [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345) https://github.com/mv-lab/swin2sr/releases.

Special thanks to the creators of the Swin2SR model and the dependencies used in this module.

