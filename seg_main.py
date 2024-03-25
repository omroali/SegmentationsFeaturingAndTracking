import os
from tqdm import tqdm

from datetime import datetime
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import get_images_in_path, show_image_list

import multiprocessing as mp

dir_path = os.path.dirname(os.path.realpath(__file__))
path = "/home/kyro/Code/SegmentationsFeaturingAndTracking/data/ball_frames"


def store_image_data(log_data, time: datetime):
    """method to store in a text file the image data for processing"""
    check_path = os.path.exists(f"process_data/{time}/data.txt")
    if not check_path:
        with open(f"process_data/{time}/data.txt", "w") as f:
            for key in log_data.keys():
                f.write(f"{log_data[key]}\n")


def process_image(inputs) -> None:
    """method to process the image"""
    [image_path, time] = inputs
    image = ImageSegmentation(image_path)
    data = image.preprocessing2(image)
    processed_images = {}
    log_data = {}
    for key in data.keys():
        if data[key]["show"] is not False:
            processed_images[key] = data[key]["image"]
            log_data[key] = f'{data[key]["operation"]} - {data[key]["params"]}'

    store_image_data(log_data, time)

    name = os.path.splitext(os.path.basename(image_path))[0]

    show_image_list(
        image_dict=processed_images,
        figsize=(10, 10),
        save_path=f"process_data/{time}/{name}",
    )


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
