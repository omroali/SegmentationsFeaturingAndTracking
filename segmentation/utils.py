import os
import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_images_and_masks_in_path(folder_path):
    images = sorted(filter(os.path.isfile, glob.glob(folder_path + "/*")))
    image_list = []
    mask_list = []
    for file_path in images:
        if "data.txt" not in file_path:
            if "GT" not in file_path:
                image_list.append(file_path)
            else:
                mask_list.append(file_path)

    return natsorted(image_list), natsorted(mask_list)


# source and modofied from https://stackoverflow.com/a/67992521
def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


from heapq import nlargest, nsmallest


def dice_score(processed_images, masks, save_path):
    eval = []
    score_dict = {}
    for idx, image in enumerate(processed_images):
        score = dice_similarity_score(image, masks[idx], save_path)
        score_dict[image] = score
        if len(eval) == 0 or max(eval) < score:
            max_score = score
            max_score_image = image
        if len(eval) == 0 or min(eval) > score:
            min_score = score
            min_score_image = image
        eval.append(score)
    avg_score = sum(eval) / len(eval)
    max_text = f"Max Score: {max_score} - {max_score_image}\n"
    min_text = f"Min Score: {min_score} - {min_score_image}\n"
    avg_text = f"Avg Score: {avg_score}\n"
    print("--- " + save_path + "\n")
    print(max_text)
    print(min_text)
    print(avg_text)
    print("---")

    FiveHighest = nlargest(5, score_dict, key=score_dict.get)
    FiveLowest = nsmallest(5, score_dict, key=score_dict.get)
    with open(f"{save_path}/dice_score.txt", "w") as f:
        f.write("---\n")
        f.write(max_text)
        f.write(min_text)
        f.write(avg_text)
        f.write("---\n")
        f.write("Scores:\n")
        for idx, score in enumerate(eval):
            f.write(f"\t{score}\t{masks[idx]}\n")
        f.write("---\n")
        f.write("5 highest:\n")
        for v in FiveHighest:
            f.write(f"{v}, {score_dict[v]}\n")
        f.write("---\n")
        f.write("5 lowest:\n")
        for v in FiveLowest:
            f.write(f"{v}, {score_dict[v]}\n")


def dice_similarity_score(seg_path, mask_path, save_path):
    """
    Computes the Dice similarity score between two binary images.
    The Dice similarity score is defined as:
    2 * |A ∩ B| / (|A| + |B|)
    where A is the first image and B is the second image.
    """

    seg = cv2.threshold(cv2.imread(seg_path), 127, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.threshold(cv2.imread(mask_path), 127, 255, cv2.THRESH_BINARY)[1]
    intersection = cv2.bitwise_and(seg, mask)
    dice_score = 2.0 * intersection.sum() / (seg.sum() + mask.sum())

    difference = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_not(seg), mask))
    cv2.imwrite(save_path + f"/difference_ds_{dice_score}.jpg", difference)
    return dice_score


def show_image_list(
    image_dict: dict = {},
    list_cmaps=None,
    grid=False,
    num_cols=2,
    figsize=(20, 10),
    title_fontsize=12,
    save_path=None,
):
    """
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images_dict: dict of image titles and image object
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    """

    list_titles, list_images = list(image_dict.keys()), list(image_dict.values())

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), "%d imgs != %d titles" % (
            len(list_images),
            len(list_titles),
        )

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), "%d imgs != %d cmaps" % (
            len(list_images),
            len(list_cmaps),
        )

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img = list_images[i]
        title = list_titles[i] if list_titles is not None else "Image %d" % (i)
        cmap = (
            list_cmaps[i]
            if list_cmaps is not None
            else (None if img_is_color(img) else "gray")
        )

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)
        list_axes[i].axis("off")

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    plt.close(fig)


def fill(img):
    des = cv2.bitwise_not(img.copy())
    contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    return cv2.bitwise_not(des)
