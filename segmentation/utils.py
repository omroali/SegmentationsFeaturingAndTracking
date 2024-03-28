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


def dice_similarity_score(segmented_image_path, mask_path):
    """
    Computes the Dice similarity score between two binary images.
    The Dice similarity score is defined as:
    2 * |A ∩ B| / (|A| + |B|)
    where A is the first image and B is the second image.

    Parameters:
    ----------
    segmented_image: Numpy array
        A binary image.
    mask: Numpy array
        A binary image.

    Returns:
    -------
    float
        The Dice similarity score.
    """

    seg = cv2.imread(segmented_image_path)
    mask = cv2.imread(mask_path)
    cv2.imshow("seg", seg)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    # assert seg.shape == mask.shape

    intersection = np.logical_and(seg, mask)
    return 2.0 * intersection.sum() / (seg.sum() + mask.sum())


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
    # _ = plt.show()

    if save_path is not None:
        fig.savefig(save_path)

    plt.close(fig)


def fillhole(input_image):
    """
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    """
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out


def fill(img):
    des = cv2.bitwise_not(img.copy())
    contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    return cv2.bitwise_not(des)


def fillCirc(img):
    des = cv2.bitwise_not(img.copy())
    contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for cnt in contour:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        # if (len(approx) > 8) & (area > 30):
        contour_list.append(cnt)
    cv2.drawContours(des, contour_list, 0, 255, -1)
    return cv2.bitwise_not(des)


def fill2(img):
    des = cv2.bitwise_not(img.copy())
    contour, hier = cv2.findContours(des, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    return cv2.bitwise_not(des)
