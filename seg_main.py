import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import math

# import segmentation
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import get_images_in_path

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

    # images = {
    #     "original": "image",
    #     "dilation": "dilation",
    #     "closing": "closing",
    # }

    rows = math.ceil(len(images) / 3)
    cols = 3
    for idx, key in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(images[key], "gray")
        plt.axis("off")
    # plt.savefig("final_image_name.extension")  # To save figure
    plt.show()  # To show figure


if __name__ == "__main__":
    main()
