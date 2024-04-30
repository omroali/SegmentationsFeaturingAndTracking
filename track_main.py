import os
import cv2

from segmentation.utils import get_images_and_masks_in_path


def evaluate_non_compactness(mask_path):
    """
    Computes the non-compactness metric of a binary mask.
    The non-compactness metric is defined as:
    1 - (4 * pi * area) / (perimeter^2)
    where area is the number of non-zero pixels in the mask and
    perimeter is the length of the mask's contour.

    Parameters:
    ----------
    mask_path: str
        The path to the binary mask.

    Returns:
    -------
    float
        The non-compactness metric.
    """

    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute the area and perimeter
    area = cv2.countNonZero(mask)
    perimeter = cv2.arcLength(contours[0], closed=True)

    # Compute the non-compactness metric
    non_compactness = 1 - (4 * 3.14159 * area) / (perimeter**2)

    # Plotting the distribution of the non-compactness metric
    # non_compactness_values = []
    # for mask_path in mask_paths:
    #     non_compactness = evaluate_non_compactness(mask_path)
    #     non_compactness_values.append(non_compactness)
    # plt.hist(non_compactness_values, bins=20)
    # plt.xlabel("Non-compactness")
    # plt.ylabel("Frequency")
    # plt.show()

    return non_compactness


def evaluate_solidity(mask_path):
    """
    Computes the solidity metric of a binary mask.
    The solidity metric is defined as:
    area / convex_area
    where area is the number of non-zero pixels in the mask and
    convex_area is the area of the convex hull of the mask.

    Parameters:
    ----------
    mask_path: str
        The path to the binary mask.

    Returns:
    -------
    float
        The solidity metric.
    """

    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute the area and convex area
    area = cv2.countNonZero(mask)
    convex_area = cv2.contourArea(contours[0])

    # Compute the solidity metric
    solidity = area / convex_area

    # plotting the distribution of the solidity metric
    # solidity_values = []
    # for mask_path in mask_paths:
    #     solidity = evaluate_solidity(mask_path)
    #     solidity_values.append(solidity)
    # plt.hist(solidity_values, bins=20)
    # plt.xlabel("Solidity")
    # plt.ylabel("Frequency")
    # plt.show()

    return solidity


def evaluate_circularity(mask_path):
    """
    Computes the circularity metric of a binary mask.
    The circularity metric is defined as:
    4 * pi * area / (perimeter^2)
    where area is the number of non-zero pixels in the mask and
    perimeter is the length of the mask's contour.

    Parameters:
    ----------
    mask_path: str
        The path to the binary mask.

    Returns:
    -------
    float
        The circularity metric.
    """

    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute the area and perimeter
    area = cv2.countNonZero(mask)
    perimeter = cv2.arcLength(contours[0], closed=True)

    # Compute the circularity metric
    circularity = (4 * 3.14159 * area) / (perimeter**2)

    # Plotting the distribution of the circularity metric
    # circularity_values = []
    # for mask_path in mask_paths:
    #     circularity = evaluate_circularity(mask_path)
    #     circularity_values.append(circularity)
    # plt.hist(circularity_values, bins=20)
    # plt.xlabel("Circularity")
    # plt.ylabel("Frequency")
    # plt.show()

    return circularity


def evaluate_eccentricity(mask_path):
    """
    Computes the eccentricity metric of a binary mask.
    The eccentricity metric is defined as:
    sqrt(1 - b^2 / a^2)
    where a and b are the major and minor axes of the mask's ellipse.

    Parameters:
    ----------
    mask_path: str
        The path to the binary mask.

    Returns:
    -------
    float
        The eccentricity metric.
    """

    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fit an ellipse to the mask
    ellipse = cv2.fitEllipse(contours[0])

    # Compute the major and minor axes
    a = max(ellipse[1])
    b = min(ellipse[1])

    # Compute the eccentricity metric
    eccentricity = (1 - (b**2) / (a**2)) ** 0.5

    # Plotting the distribution of the eccentricity metric
    # eccentricity_values = []
    # for mask_path in mask_paths:
    #     eccentricity = evaluate_eccentricity(mask_path)
    #     eccentricity_values.append(eccentricity)
    # plt.hist(eccentricity_values, bins=20)
    # plt.xlabel("Eccentricity")
    # plt.ylabel("Frequency")
    # plt.show()

    return eccentricity


""" (texture features) Calculate the normalised grey-level co-occurrence matrix in four
orientations (0°, 45°, 90°, 135°) for the patches from the three balls, separately for each of the
colour channels (red, green, blue). For each orientation, calculate the first three features
proposed by Haralick et al. (Angular Second Moment, Contrast, Correlation), and produce per-
patch features by calculating the feature average and range across the 4 orientations. Select
one feature from each of the colour channels and plot the distribution per ball type. """


def texture_features(img, mask):
    # Load the image
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the mask to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Find the contours of the mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a mask for the region of interest
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    # Extract the region of interest
    roi = cv2.bitwise_and(gray, mask)
    # Define the patch size
    patch_size = 32
    # Extract patches from the region of interest
    patches = []
    for i in range(0, roi.shape[0], patch_size):
        for j in range(0, roi.shape[1], patch_size):
            patch = roi[i : i + patch_size, j : j + patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
    # Define the co-occurrence matrix parameters
    distances = [1]
    angles = [0, 45, 90, 135]
    levels = 256
    symmetric = True
    normed = True
    # Calculate the co-occurrence matrix for each patch
    features = []
    for patch in patches:
        glcm = greycomatrix(
            patch, distances, angles, levels, symmetric=symmetric, normed=normed
        )
        # Calculate the Haralick features
        asm = greycoprops(glcm, "ASM").flatten()
        contrast = greycoprops(glcm, "contrast").flatten()
        correlation = greycoprops(glcm, "correlation").flatten()
        # Calculate the feature average and range across the 4 orientations
        asm_avg = np.mean(asm)
        contrast_avg = np.mean(contrast)
        correlation_avg = np.mean(correlation)
        asm_range = np.ptp(asm)
        contrast_range = np.ptp(contrast)
        correlation_range = np.ptp(correlation)
        features.append(
            [
                asm_avg,
                contrast_avg,
                correlation_avg,
                asm_range,
                contrast_range,
                correlation_range,
            ]
        )
    # Convert the features to a NumPy array
    features = np.array(features)
    # Select one feature from each of the colour channels
    red_features = features[:, 0]
    green_features = features[:, 1]
    blue_features = features[:, 2]
    # Plot the distribution per ball type
    plt.hist(red_features, bins=20, color="red", alpha=0.5, label="Red")
    plt.hist(green_features, bins=20, color="green", alpha=0.5, label="Green")
    plt.hist(blue_features, bins=20, color="blue", alpha=0.5, label="Blue")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    return features


if __name__ == "__main__":

    # get images and masks in path
    images, masks = get_images_and_masks_in_path("data/ball_frames")

    # Load a binary image (you can replace this with your own binary image)
    img = cv2.imread("data/ball_frames/frame-100_GT.png", cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # overlay binay image over it's rgb counterpart
    image = cv2.imread("data/ball_frames/frame-100.png")
    image = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), image)

    print(
        "Non-compactness:",
        evaluate_non_compactness("data/ball_frames/frame-100_GT.png"),
    )
    print("Solidity:", evaluate_solidity("data/ball_frames/frame-100_GT.png"))
    print("Circularity:", evaluate_circularity("data/ball_frames/frame-100_GT.png"))
    print("Eccentricity:", evaluate_eccentricity("data/ball_frames/frame-100_GT.png"))

    cv2.imshow("Overlay", image)
    cv2.waitKey(0)
