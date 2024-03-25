import os
from tqdm import tqdm

from datetime import datetime
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import get_images_in_path, show_image_list

import multiprocessing as mp

dir_path = os.path.dirname(os.path.realpath(__file__))
path = "/home/kyro/Code/SegmentationsFeaturingAndTracking/data/ball_frames"


def store_image_data(image: ImageSegmentation, time: datetime):
    """method to store in a text file the image data for processing"""
    check_path = os.path.exists(f"process_data/{time}/data.txt")
    if not check_path:
        with open(f"process_data/{time}/data.txt", "w") as f:
            for entry in image.processing_data:
                f.write(f"{entry}\n")


def process_image(inputs) -> None:
    """method to process the image"""
    [image_path, time] = inputs
    image = ImageSegmentation(image_path)
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

    name = os.path.splitext(os.path.basename(image_path))[0]

    show_image_list(
        image_dict=images,
        figsize=(10, 10),
        save_path=f"process_data/{time}/{name}",
    )

    store_image_data(image, time)


def main():
    images = get_images_in_path(path)
    time = datetime.now().isoformat("_", timespec="seconds")
    os.mkdir(f"process_data/{time}")

    with mp.Pool(processes=10) as pool:
        inputs = [[image, time] for image in images]
        list(
            tqdm(
                pool.imap_unordered(process_image, inputs, chunksize=4),
                total=len(images),
            )
        )
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
