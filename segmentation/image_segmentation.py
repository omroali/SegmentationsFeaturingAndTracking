import cv2
import numpy as np
from segmentation.utils import fillhole, fill, fillCirc


class ImageSegmentation:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.preprocessing()

    def preprocessing(self):
        """grayscaling and smoothing of the images"""
        # extracting the intensity value of the image
        self.grayscale = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGRA2GRAY)
        self.hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
        (self.hue, self.saturation, self.intensity) = cv2.split(self.hsv)
        # -- analysis opportunity - intensity vs grayscale --

        # applying gaussian blurring
        Kv = 5
        blur = cv2.GaussianBlur(self.intensity, (Kv, Kv), cv2.BORDER_DEFAULT)
        # blur = cv2.GaussianBlur(self.grayscale, (Kv, Kv), cv2.BORDER_DEFAULT)

        # applying an adaptive thresholding function
        # -- analysis opportunity - impact of different thresholding --
        # self.adaptive_mean_threshold = cv2.adaptiveThreshold(
        #     src=blur,
        #     maxValue=255,
        #     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #     thresholdType=cv2.THRESH_BINARY,
        #     blockSize=11,
        #     C=2,
        # )
        self.adaptive_gaussian_threshold = cv2.adaptiveThreshold(
            src=blur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=5,
        )

        Kc = 5
        self.dilation = cv2.dilate(
            self.adaptive_gaussian_threshold.copy(), (Kc, Kc), iterations=7
        )

        # noise reduction in a binary image https://www.researchgate.net/publication/4376361_Noise_removal_and_enhancement_of_binary_images_using_morphological_operations

        self.closing = cv2.morphologyEx(
            self.adaptive_gaussian_threshold.copy(),
            cv2.MORPH_CLOSE,
            (Kc, Kc),
            iterations=10,
        )

        self.opening = cv2.morphologyEx(
            self.adaptive_gaussian_threshold.copy(),
            cv2.MORPH_OPEN,
            (Kc, Kc),
            iterations=10,
        )

        self.dialte_opening = cv2.morphologyEx(
            self.dilation.copy(),
            cv2.MORPH_OPEN,
            (Kc, Kc),
            iterations=7,
        )
        self.dialte_closing = cv2.morphologyEx(
            self.dilation.copy(),
            cv2.MORPH_CLOSE,
            (Kc, Kc),
            iterations=7,
        )
        # h, w = self.opening.shape[:2]
        # im_floodfill = self.opening.copy()
        # mask = np.zeros((h + 2, w + 2), np.uint8)
        #
        # self.imfill = cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        #
        # Combine the two images to get the foreground.
        # self.im_out = self.opening | im_floodfill_inv
        #
        # des = cv2.bitwise_not(self.closing.copy())
        # contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # #
        # for cnt in contour:
        #     cv2.drawContours(des, [cnt], 0, 255, -1)
        # #
        # self.fill = cv2.bitwise_not(des)

        self.fill = fill(self.closing.copy())
        self.fill2 = fill(self.opening.copy())
        self.fill3 = fillCirc(self.opening.copy())
        # self.fill3 = fill(self.open_close.copy())
        # self.fll2 = fillhole(self.opening)
