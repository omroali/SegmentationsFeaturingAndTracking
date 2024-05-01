import os
import cv2

from cv2.gapi import bitwise_and
from matplotlib import pyplot as plt
from matplotlib.artist import get

from segmentation.utils import get_images_and_masks_in_path
import numpy as np
from segmentation.utils import fill
import math
from skimage.measure import regionprops


def shape_features_eval(contour):
    area = cv2.contourArea(contour)
    # props = regionprops(contour)

    # getting non-compactness
    perimeter = cv2.arcLength(contour, closed=True)
    non_compactness = 1 - (4 * math.pi * area) / (perimeter**2)

    # getting solidity
    convex_hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(convex_hull)

    solidity = area / convex_area
    # solidity = props[0].solidity
    # print(props)

    # getting circularity
    circularity = (4 * math.pi * area) / (perimeter**2)

    # getting eccentricity
    ellipse = cv2.fitEllipse(contour)
    a = max(ellipse[1])
    b = min(ellipse[1])
    eccentricity = (1 - (b**2) / (a**2)) ** 0.5

    return {
        "non_compactness": non_compactness,
        "solidity": solidity,
        "circularity": circularity,
        "eccentricity": eccentricity,
    }


from skimage.feature import graycomatrix, graycoprops


def texture_features_eval(patch):
    # # Define the co-occurrence matrix parameters
    distances = [1]
    angles = np.radians([0, 45, 90, 135])
    levels = 256
    symmetric = True
    normed = True
    glcm = graycomatrix(
        patch, distances, angles, levels, symmetric=symmetric, normed=normed
    )
    filt_glcm = glcm[1:, 1:, :, :]

    # Calculate the Haralick features
    asm = graycoprops(filt_glcm, "ASM").flatten()
    # contrast = graycoprops(glcm, "contrast").flatten()
    contrast = graycoprops(filt_glcm, "contrast").flatten()
    correlation = graycoprops(filt_glcm, "correlation").flatten()

    # Calculate the feature average and range across the 4 orientations
    asm_avg = np.mean(asm)
    contrast_avg = np.mean(contrast)
    correlation_avg = np.mean(correlation)
    asm_range = np.ptp(asm)
    contrast_range = np.ptp(contrast)
    correlation_range = np.ptp(correlation)

    return {
        "asm": asm,
        "contrast": contrast,
        "correlation": correlation,
        "asm_avg": asm_avg,
        "contrast_avg": contrast_avg,
        "correlation_avg": correlation_avg,
        "asm_range": asm_range,
        "contrast_range": contrast_range,
        "correlation_range": correlation_range,
    }


def initialise_channels_features():
    def initialise_channel_texture_features():
        return {
            "asm": [],
            "contrast": [],
            "correlation": [],
            "asm_avg": [],
            "contrast_avg": [],
            "correlation_avg": [],
            "asm_range": [],
            "contrast_range": [],
            "correlation_range": [],
        }

    return {
        "blue": initialise_channel_texture_features(),
        "green": initialise_channel_texture_features(),
        "red": initialise_channel_texture_features(),
    }


def initialise_shape_features():
    return {
        "non_compactness": [],
        "solidity": [],
        "circularity": [],
        "eccentricity": [],
    }


def get_all_features_balls(path):
    features = {
        "football": {
            "shape_features": initialise_shape_features(),
            "texture_features": initialise_channels_features(),
        },
        "soccer": {
            "shape_features": initialise_shape_features(),
            "texture_features": initialise_channels_features(),
        },
        "tennis": {
            "shape_features": initialise_shape_features(),
            "texture_features": initialise_channels_features(),
        },
    }

    images, masks = get_images_and_masks_in_path(path)
    for idx, _ in enumerate(images):
        image = images[idx]
        mask = masks[idx]
        msk = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        _, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)

        # overlay binay image over it's rgb counterpart
        img = cv2.imread(image)
        img = cv2.bitwise_and(cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR), img)
        contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)
            ball_img = np.zeros(msk.shape, dtype=np.uint8)
            cv2.drawContours(ball_img, contour, -1, (255, 255, 255), -1)
            fill_img = cv2.bitwise_not(fill(cv2.bitwise_not(ball_img)))
            rgb_fill = cv2.bitwise_and(cv2.cvtColor(fill_img, cv2.COLOR_GRAY2BGR), img)

            out = fill_img.copy()
            out_colour = rgb_fill.copy()

            # Now crop image to ball size
            (y, x) = np.where(fill_img == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            padding = 3
            out = out[
                topy - padding : bottomy + padding, topx - padding : bottomx + padding
            ]
            out_colour = out_colour[
                topy - padding : bottomy + padding, topx - padding : bottomx + padding
            ]

            # getting ball features
            shape_features = shape_features_eval(contour)
            texture_features_colour = {
                "blue": texture_features_eval(out_colour[:, :, 0]),
                "green": texture_features_eval(out_colour[:, :, 1]),
                "red": texture_features_eval(out_colour[:, :, 2]),
            }

            # segmenting ball by using area
            if area > 1300:  # football
                append_ball = "football"
            elif area > 500:  # soccer_ball
                append_ball = "soccer"
            else:  # tennis ball
                append_ball = "tennis"

            for key in shape_features:
                features[append_ball]["shape_features"][key].append(shape_features[key])

            for colour in texture_features_colour.keys():
                for colour_feature in texture_features_colour[colour]:
                    features[append_ball]["texture_features"][colour][
                        colour_feature
                    ].append(texture_features_colour[colour][colour_feature])
    return features


def feature_stats(features, ball, colours=["blue", "green", "red"]):
    def get_stats(array):
        return {
            "mean": np.mean(array),
            "std": np.std(array),
            "min": np.min(array),
            "max": np.max(array),
        }

    def get_ball_shape_stats(features, ball):
        feature_find = ["non_compactness", "solidity", "circularity", "eccentricity"]
        return {
            feature: get_stats(features[ball]["shape_features"][feature])
            for feature in feature_find
        }

    def get_ball_texture_stats(features, ball, colour):
        feature_find = ["asm_avg", "contrast_avg", "correlation_avg"]
        return {
            texture: get_stats(features[ball]["texture_features"][colour][texture])
            for texture in feature_find
        }

    stats = {
        ball: {
            "shape_features": get_ball_shape_stats(features, ball),
            "texture_features": {
                colour: get_ball_texture_stats(features, ball, colour)
                for colour in colours
            },
        },
    }
    return stats


def get_histogram(data, Title):
    """
    data {ball: values}
    """
    plt.figure()
    for ball, values in data.items():
        plt.hist(values, bins=10, alpha=0.5, label=ball)
    plt.title(Title + " Histogram")
    plt.xlabel(Title)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    features = get_all_features_balls("data/ball_frames")
    # stats = feature_stats(features, "football", ["red", "green", "blue"])
    # print(stats)

    balls = [
        "tennis",
        "soccer",
        "football",
    ]

    non_compactness = {
        ball: features[ball]["shape_features"]["non_compactness"] for ball in balls
    }
    solidity = {ball: features[ball]["shape_features"]["solidity"] for ball in balls}
    # print(solidity)
    circularity = {
        ball: features[ball]["shape_features"]["circularity"] for ball in balls
    }
    eccentricity = {
        ball: features[ball]["shape_features"]["eccentricity"] for ball in balls
    }

    get_histogram(non_compactness, "Non-Compactness")
    get_histogram(solidity, "Soliditiy")
    get_histogram(circularity, "Circularity")
    get_histogram(eccentricity, "Eccentricity")

    # TODO: calculate range of the feature average

    channel_colours = ["red", "green", "blue"]

    def get_ch_features(feature_name):
        return {
            colour: {
                ball: features[ball]["texture_features"][colour][feature_name]
                for ball in balls
            }
            for colour in channel_colours
        }

    asm_avg = get_ch_features("asm_avg")
    contrast_avg = get_ch_features("contrast_avg")
    correlation_avg = get_ch_features("correlation_avg")

    red_asm_data = [asm_avg["red"][ball] for ball in balls]
    green_asm_data = [asm_avg["green"][ball] for ball in balls]
    blue_asm_data = [asm_avg["blue"][ball] for ball in balls]

    red_contrast_data = [contrast_avg["red"][ball] for ball in balls]
    green_contrast_data = [contrast_avg["green"][ball] for ball in balls]
    blue_contrast_data = [contrast_avg["blue"][ball] for ball in balls]

    red_correlation_data = [correlation_avg["red"][ball] for ball in balls]
    green_correlation_data = [correlation_avg["green"][ball] for ball in balls]
    blue_correlation_data = [correlation_avg["blue"][ball] for ball in balls]

    asm_data = [red_asm_data, green_asm_data, blue_asm_data]
    contrast_data = [red_contrast_data, green_contrast_data, blue_contrast_data]
    correlation_data = [
        red_correlation_data,
        green_correlation_data,
        blue_correlation_data,
    ]

    asm_titles = ["R-ASM", "G-ASM", "B-ASM"]
    contrast_titles = ["R-Contrast", "G-Contrast", "B-Contrast"]
    correlation_titles = ["R-Correlation", "G-Correlation", "B-Correlation"]

    plt_colours = ["yellow", "white", "orange"]
    plt.figure()
    for i, d in enumerate(asm_data):
        plt.subplot(3, 3, i + 1)
        box = plt.boxplot(d, patch_artist=True, widths=0.2)
        plt.xticks([1, 2, 3], balls)
        plt.title(asm_titles[i])
        for j, patch in enumerate(box["boxes"]):
            patch.set_facecolor(plt_colours[j])
    for i, d in enumerate(contrast_data):
        plt.subplot(3, 3, i + 3 + 1)
        box = plt.boxplot(d, patch_artist=True, widths=0.2)
        plt.xticks([1, 2, 3], balls)
        plt.title(contrast_titles[i])
        for j, patch in enumerate(box["boxes"]):
            patch.set_facecolor(plt_colours[j])
    for i, d in enumerate(correlation_data):
        plt.subplot(3, 3, i + 6 + 1)
        box = plt.boxplot(d, patch_artist=True, widths=0.2)
        plt.xticks([1, 2, 3], balls)
        plt.title(correlation_titles[i])
        for j, patch in enumerate(box["boxes"]):
            patch.set_facecolor(plt_colours[j])
    plt.tight_layout()
    plt.show()
