import cv2


class ImageSegmentation:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.preprocessing()

    def preprocessing(self):
        """grayscaling and smoothing of the images"""
        # extracting the intensity value of the image
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        (self.hue, self.saturation, self.intensity) = cv2.split(self.hsv)
        # -- analysis opportunity - intensity vs grayscale --

        # applying gaussian blurring
        Kv = 5
        blur = cv2.GaussianBlur(self.intensity, (Kv, Kv), cv2.BORDER_DEFAULT)

        # applying an adaptive thresholding function
        # -- analysis opportunity - impact of different thresholding --
        self.adaptive_mean_threshold = cv2.adaptiveThreshold(
            src=blur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        self.adaptive_gaussian_threshold = cv2.adaptiveThreshold(
            src=blur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=5,
        )

        Kc = 5
        self.dilation = cv2.dilate(
            self.adaptive_gaussian_threshold, (Kc, Kc), iterations=9
        )
        self.closing = cv2.morphologyEx(
            self.dilation, cv2.MORPH_OPEN, (Kc, Kc), iterations=9
        )
