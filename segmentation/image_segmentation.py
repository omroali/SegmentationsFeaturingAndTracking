import cv2
import numpy as np
from segmentation.utils import fillhole, fill, fillCirc


class ImageSegmentation:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)

        self.processing_data = []
        # self.data = [
        #     "Kblur": (3, 3),
        #     "blockSize": 15,
        #     "C": 3,
        #     "KDil": (5, 5),
        #     "iDil": 7,
        #     "KClos": (5, 5),
        #     "iClos": 10,
        #     "KOpen": (5, 5),
        #     "iOpen": 7,
        # ]

        # self.preprocessing()

    def blur(self, image, ksize=(3, 3), iterations=1):
        """apply gaussian blur to the image"""
        self.processing_data.append(
            [f"operation,gblur,kernel:{ksize},iterations:{iterations}"]
        )

        self.blur = image.copy()

        for _ in range(iterations):
            self.blur = cv2.GaussianBlur(self.blur, ksize, cv2.BORDER_DEFAULT)
        return self.blur

    def adaptive_threshold(self, image, blockSize=15, C=3):
        """apply adaptive threshold to the image"""
        self.processing_data.append(
            [f"operation:adaptive_threshold,blockSize:{blockSize},C:{C}"]
        )
        image = image.copy()

        self.adaptive_gaussian_threshold = cv2.adaptiveThreshold(
            src=image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,
            C=C,
        )
        return self.adaptive_gaussian_threshold

    def dilate(self, image, kernel=(5, 5), iterations=7):
        """apply dilation to the image"""
        self.processing_data.append(
            [f"operation:dilate,kernel:{kernel},iterations:{iterations}"]
        )
        image = image.copy()

        self.dilation = cv2.dilate(
            src=image,
            kernel=kernel,
            iterations=iterations,
        )
        return self.dilation

    def closing(self, image, kernel=(5, 5), iterations=10):
        """apply closing to the image"""
        self.processing_data.append(
            [f"operation:close,kernel:{kernel},iterations:{iterations}"]
        )
        image = image.copy()

        self.closing = cv2.morphologyEx(
            src=image,
            op=cv2.MORPH_CLOSE,
            kernel=kernel,
            iterations=iterations,
        )
        return self.closing

    def opening(self, image, kernel=(5, 5), iterations=7):
        """apply opening to the image"""
        self.processing_data.append(
            [f"operation:open,kernel:{kernel},iterations:{iterations}"]
        )
        image = image.copy()

        self.opening = cv2.morphologyEx(
            src=image,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
            iterations=iterations,
        )
        return self.opening

    # # @staticmethod
    # def preprocessing(self):
    #     """grayscaling and smoothing of the images"""
    #     # extracting the intensity value of the image
    #     self.grayscale = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGRA2GRAY)
    #     self.hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
    #     (self.hue, self.saturation, self.intensity) = cv2.split(self.hsv)
    #     # -- analysis opportunity - intensity vs grayscale --
    #
    #     # applying gaussian blurring
    #     blur = cv2.GaussianBlur(self.intensity, self.data["Kblur"], cv2.BORDER_DEFAULT)
    #     for i in range(3):
    #         blur = cv2.GaussianBlur(blur, self.data["Kblur"], cv2.BORDER_DEFAULT)
    #
    #     self.adaptive_gaussian_threshold = cv2.adaptiveThreshold(
    #         src=blur,
    #         maxValue=255,
    #         adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #         thresholdType=cv2.THRESH_BINARY,
    #         blockSize=self.data["blockSize"],
    #         C=self.data["C"],
    #     )
    #
    #     self.dilation = cv2.dilate(
    #         src=self.adaptive_gaussian_threshold.copy(),
    #         kernel=self.data["KDil"],
    #         iterations=self.data["iDil"],
    #     )
    #
    #     # noise reduction in a binary image https://www.researchgate.net/publication/4376361_Noise_removal_and_enhancement_of_binary_images_using_morphological_operations
    #
    #     self.closing = cv2.morphologyEx(
    #         src=self.adaptive_gaussian_threshold.copy(),
    #         op=cv2.MORPH_CLOSE,
    #         kernel=self.data["KClos"],
    #         iterations=self.data["iClos"],
    #     )
    #
    #     self.opening = cv2.morphologyEx(
    #         src=self.adaptive_gaussian_threshold.copy(),
    #         op=cv2.MORPH_OPEN,
    #         kernel=self.data["KOpen"],
    #         iterations=self.data["iOpen"],
    #     )
    #
    #     K = 5
    #     iterations = 7
    #     self.dialte_opening = cv2.morphologyEx(
    #         src=self.dilation.copy(),
    #         op=cv2.MORPH_OPEN,
    #         kernel=(K, K),
    #         iterations=iterations,
    #     )
    #     self.dialte_closing = cv2.morphologyEx(
    #         src=self.dilation.copy(),
    #         op=cv2.MORPH_CLOSE,
    #         kernel=(K, K),
    #         iterations=iterations,
    #     )
    #
    #     self.fill = fill(self.closing.copy())
    #     self.fill2 = fill(self.opening.copy())
    #     self.fill3 = fillCirc(self.fill2.copy())
    #     # self.fill3 = fill(self.open_close.copy())
    #     # self.fll2 = fillhole(self.opening)

    # @staticmethod
    def preprocessing2(self,image):
        image.grayscale = cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY)
        image.hsv = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        (image.hue, image.saturation, image.intensity) = cv2.split(image.hsv)

        image.blur(image.intensity, ksize=(3, 3), iterations=5)
        image.adaptive_threshold(
            image=image.blur,
            blockSize=15,
            C=3,
        )
        image.dilate(
            image.adaptive_gaussian_threshold,
            kernel=(5, 5),
            iterations=7,
        )
        image.closing(
            image.adaptive_gaussian_threshold,
            kernel=(5, 5),
            iterations=10,
        )
        image.opening(
            image.adaptive_gaussian_threshold,
            kernel=(5, 5),
            iterations=7,
        )
        # image.dialte_opening(image.dilation, kernel=(5, 5), iterations=7)
        # image.dialte_closing(image.dilation, kernel=(5, 5), iterations=7)
        image.fill = fill(image.closing.copy())
        image.fill2 = fill(image.opening.copy())
        image.fill3 = fillCirc(image.fill2.copy())
