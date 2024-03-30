import cv2
import numpy as np


def morphological_filters(img, operation, structuring_element, iterations):
    # Check if the input image is binary
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError(
            "Input image must be a binary image (2D numpy array of dtype uint8)."
        )

    # Check if the operation is valid
    valid_operations = ["open", "close", "erode", "dilate"]
    if operation not in valid_operations:
        raise ValueError(
            "Invalid morphological operation. Valid operations are:", valid_operations
        )

    # Check if the structuring element is valid
    if structuring_element not in ["rect", "cross", "ellipse"]:
        raise ValueError(
            "Invalid structuring element. Valid elements are:",
            ["rect", "cross", "ellipse"],
        )

    # Check if the number of iterations is a positive integer
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("Number of iterations must be a positive integer.")

    # Create the structuring element
    if structuring_element == "rect":
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    elif structuring_element == "cross":
        structuring_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    elif structuring_element == "ellipse":
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Apply the morphological operation
    if operation == "open":
        result = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, structuring_element, iterations=iterations
        )
    elif operation == "close":
        result = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, structuring_element, iterations=iterations
        )
    elif operation == "erode":
        result = cv2.morphologyEx(
            img, cv2.MORPH_ERODE, structuring_element, iterations=iterations
        )
    elif operation == "dilate":
        result = cv2.morphologyEx(
            img, cv2.MORPH_DILATE, structuring_element, iterations=iterations
        )

    return result


# Load a binary image (you can replace this with your own binary image)
img = cv2.imread("data/ball_frames/frame-100_GT.png", cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Apply the morphological filters
operation = "close"  # or 'close', 'erode', 'dilate'
structuring_element = "rect"  # or 'cross', 'ellipse'
iterations = 10  # Number of iterations
result = morphological_filters(img, operation, structuring_element, iterations)

# Display the result
cv2.imshow("Morphological Filters", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
