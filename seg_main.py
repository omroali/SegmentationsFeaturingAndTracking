import os
from tqdm import tqdm

from datetime import datetime
from segmentation.image_segmentation import ImageSegmentation
from segmentation.utils import get_images_in_path, show_image_list

import multiprocessing as mp

dir_path = os.path.dirname(os.path.realpath(__file__))
path = "data/ball_frames"


def store_image_data(log_data, time: datetime):
    """method to store in a text file the image data for processing"""
    check_path = os.path.exists(f"process_data/{time}/data.txt")
    if not check_path:
        with open(f"process_data/{time}/data.txt", "w") as f:
            for key in log_data.keys():
                f.write(f"{log_data[key]}\n")


def process_image(inputs: list[list, bool]) -> None:
    """method to process the image"""
    [image_path, save, time] = inputs
    image = ImageSegmentation(image_path)
    data = image.preprocessing2(image)
    processed_images = {}
    log_data = {}
    for key in data.keys():
        if data[key]["show"] is not False:
            processed_images[key] = data[key]["image"]
            log_data[key] = f'{data[key]["operation"]} - {data[key]["params"]}'

    name = os.path.splitext(os.path.basename(image_path))[0]

    save_path = None
    if save:
        dir = f"process_data/{time}"
        save_path = f"{dir}/{name}"
        if not os.path.exists(dir):
            os.mkdir(dir)
        store_image_data(log_data, time)

    show_image_list(
        image_dict=processed_images,
        figsize=(10, 10),
        save_path=save_path,
    )


# def evaluating_c():
#     images = get_images_in_path(path)
#
#     time = datetime.now().isoformat("_", timespec="seconds") + f"C_{c}"
#
#     with mp.Pool() as pool:
#         inputs = [[image, time, save, c] for image in images]
#         list(
#             tqdm(
#                 pool.imap_unordered(process_image, inputs, chunksize=4),
#                 total=len(images),
#             )
#         )
#         pool.close()
#         pool.join()


def all_images(images, save=False):
    with mp.Pool() as pool:
        time = datetime.now().isoformat("_", timespec="seconds")
        inputs = [[image, save, time] for image in images]
        list(
            tqdm(
                pool.imap_unordered(process_image, inputs, chunksize=4),
                total=len(images),
            )
        )
        pool.close()
        pool.join()


def main():
    images = get_images_in_path(path)
    all_images(images, True)
    # process_image([images[10], False])


if __name__ == "__main__":
    main()
