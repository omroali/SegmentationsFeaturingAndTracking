# import cv2
import os

# from matplotlib import pyplot as plt

# import numpy as np
# import math
from tqdm import tqdm

# import segmentation
from datetime import datetime
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import get_images_in_path, show_image_list

dir_path = os.path.dirname(os.path.realpath(__file__))
path = "/home/kyro/Code/SegmentationsFeaturingAndTracking/data/ball_frames"


def main():
    images = get_images_in_path(path)
    time = datetime.now().isoformat("_", timespec="seconds")
    os.mkdir(f"process_data/{time}")
    for i, image in tqdm(enumerate(images), desc="Processing..:"):
        image = ImageSegmentation(image)

        images = {
            "original": image.image,
            "intensity": image.intensity,
            "adaptive_gaussian_threshold": image.adaptive_gaussian_threshold,
            "dilation": image.dilation,
            "closing": image.closing,
            "opening": image.opening,
            "floodfill_closing": image.fill,
            "floodfill_opening": image.fill2,
            "dialte_closing": image.dialte_closing,
            "dialte_opening": image.dialte_opening,
            "circular_fill": image.fill3,
        }

        # print(images)
        show_image_list(
            image_dict=images,
            figsize=(10, 10),
            save_path=f"process_data/{time}/img_{i}kv_3_kc_5_its_10",
        )


if __name__ == "__main__":
    main()
