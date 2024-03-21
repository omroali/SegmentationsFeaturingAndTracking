import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import math

# import segmentation
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import get_images_in_path, show_image_list

dir_path = os.path.dirname(os.path.realpath(__file__))
path = "/home/kyro/Code/SegmentationsFeaturingAndTracking/data/ball_frames"


def main():
    images = get_images_in_path(path)
    first = images[0]
    image = ImageSegmentation(first)

    images = {
        "original": image.image,
        "dilation": image.dilation,
        "closing": image.closing,
    }

    show_image_list(images, figsize=(10, 10))
   
if __name__ == "__main__":
    main()
