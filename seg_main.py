import os
import cv2
from tqdm import tqdm

from datetime import datetime
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import (
    dice_score,
    get_images_and_masks_in_path,
    show_image_list,
)

import multiprocessing as mp

dir_path = os.path.dirname(os.path.realpath(__file__))
path = "data/ball_frames"


def store_image_data(log_data, time: datetime):
    """method to store in a text file the image data for processing"""
    check_path = os.path.exists(f"process_data/{time}/data.txt")
    if not check_path:
        with open(f"process_data/{time}/data.txt", "w") as f:
            for log in log_data:
                f.write(f"{log}\n")


def process_image(inputs: list[list, bool]) -> None:
    """method to process the image"""
    [image_path, save, time, save_dir] = inputs
    image = ImageSegmentation(image_path, save_dir)
    data = image.preprocessing(image)
    processed_images = {}
    for key in data.keys():
        if data[key]["show"] is not False:
            processed_images[key] = data[key]["image"]
    log_data = image.processing_data

    name = os.path.splitext(os.path.basename(image_path))[0]

    save_path = None
    if save:
        save_path = f"{save_dir}/{name}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        store_image_data(log_data, time)

        if data["segmentation"]["image"] is not None:
            segmentation_path = f"{save_dir}/segmentation/"
            if not os.path.exists(segmentation_path):
                os.mkdir(segmentation_path)
            seg_path = f"{segmentation_path}{os.path.basename(image.image_path)}"
            cv2.imwrite(seg_path, data["segmentation"]["image"])

    show_image_list(
        image_dict=processed_images,
        figsize=(10, 10),
        save_path=save_path,
    )


def process_all_images(images, save=False):
    time = datetime.now().isoformat("_", timespec="seconds")
    save_path = f"process_data/{time}"
    seg_path = f"{save_path}/segmentation"

    with mp.Pool() as pool:
        inputs = [[image, save, time, save_path] for image in images]
        list(
            tqdm(
                pool.imap_unordered(process_image, inputs, chunksize=4),
                total=len(images),
            )
        )
        pool.close()
        pool.join()

    return save_path, seg_path


def main():
    images, masks = get_images_and_masks_in_path(path)
    processed_image_path, seg_path = process_all_images(images, True)
    processed_images, _ = get_images_and_masks_in_path(seg_path)
    dice_score(processed_images, masks, seg_path)


if __name__ == "__main__":
    main()
