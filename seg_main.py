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


# def store_image_data(image: ImageSegmentation, time: datetime):
#     """method to store in a text file the image data for processing"""
#     check_path = os.path.exists(f"process_data/{time}/data.txt")
#     if not check_path:
#         with open(f"process_data/{time}/data.txt", "w") as f:
#             f.write(f"Kblur: {image.data['Kblur']}")
#             f.write(f"blockSize: {image.data['blockSize']}")
#             f.write(f"C: {image.data['C']}")
#             f.write(f"KDil: {image.data['KDil']}")
#             f.write(f"iDil: {image.data['iDil']}")
#             f.write(f"KClos: {image.data['KClos']}")
#             f.write(f"iClos: {image.data['iClos']}")
#             f.write(f"KOpen: {image.data['KOpen']}")
#             f.write(f"iOpen: {image.data['iOpen']}")
#             f.write("\n\n")


def store_image_data(image: ImageSegmentation, time: datetime):
    """method to store in a text file the image data for processing"""
    check_path = os.path.exists(f"process_data/{time}/data.txt")
    if not check_path:
        with open(f"process_data/{time}/data.txt", "w") as f:
            for entry in image.processing_data:
                f.write(f"{entry}\n")


def main():
    images = get_images_in_path(path)
    time = datetime.now().isoformat("_", timespec="seconds")
    os.mkdir(f"process_data/{time}")

    for i, image in tqdm(enumerate(images), desc="Processing..:"):
        image = ImageSegmentation(image)
        image.preprocessing2(image)
        images = {
            "original": image.image,
            "intensity": image.intensity,
            "adaptive_gaussian_threshold": image.adaptive_gaussian_threshold,
            "dilation": image.dilation,
            "closing": image.closing,
            "opening": image.opening,
            "floodfill_closing": image.fill,
            "floodfill_opening": image.fill2,
            # "dialte_closing": image.dialte_closing,
            # "dialte_opening": image.dialte_opening,
            "circular_fill": image.fill3,
        }

        # print(images)
        show_image_list(
            image_dict=images,
            figsize=(10, 10),
            save_path=f"process_data/{time}/img_{i}",
        )

    store_image_data(image, time)


if __name__ == "__main__":
    main()
