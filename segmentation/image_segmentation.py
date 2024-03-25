import cv2
from cv2.typing import MatLike
import numpy as np
from segmentation.utils import fillhole, fill, fillCirc


class ImageSegmentation:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.processing_data = []
        self.processing_images = []

    def log_image_processing(self, image, operation: str):
        """log the image processing"""
        self.processing_data.append(operation)
        self.processing_images.append(image)

    def blur(self, image, ksize=(3, 3), iterations=1):
        """apply gaussian blur to the image"""
        blur = image.copy()
        for _ in range(iterations):
            blur = cv2.GaussianBlur(blur, ksize, cv2.BORDER_DEFAULT)
        self.log_image_processing(blur, f"gblur,kernel:{ksize},iterations:{iterations}")
        return blur

    def adaptive_threshold(self, image, blockSize=15, C=3):
        """apply adaptive threshold to the image"""
        image = image.copy()
        adaptive_gaussian_threshold = cv2.adaptiveThreshold(
            src=image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,
            C=C,
        )
        self.log_image_processing(
            adaptive_gaussian_threshold,
            f"adaptive_threshold,blockSize:{blockSize},C:{C}",
        )
        return adaptive_gaussian_threshold

    def dilate(self, image, kernel=(5, 5), iterations=7):
        """apply dilation to the image"""
        image = image.copy()
        dilation = cv2.dilate(
            src=image,
            kernel=kernel,
            iterations=iterations,
        )

        self.log_image_processing(
            dilation,
            f"dilate,kernel:{kernel},iterations:{iterations}",
        )
        return dilation

    def closing(self, image, kernel=(5, 5), iterations=10):
        """apply closing to the image"""
        image = image.copy()
        closing = cv2.morphologyEx(
            src=image,
            op=cv2.MORPH_CLOSE,
            kernel=kernel,
            iterations=iterations,
        )

        self.log_image_processing(
            closing,
            f"closing,kernel:{kernel},iterations:{iterations}",
        )
        return closing

    def opening(self, image, kernel=(5, 5), iterations=7):
        """apply opening to the image"""
        image = image.copy()
        opening = cv2.morphologyEx(
            src=image,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
            iterations=iterations,
        )
        self.log_image_processing(
            opening,
            f"opening,kernel:{kernel},iterations:{iterations}",
        )
        return opening

    @staticmethod
    def preprocessing2(image):
        return_data = {}

        return_data["original"] = {
            "image": image.image,
            "operation": "original",
            "params": "",
            "show": True,
        }
        return_data["grayscale"] = {
            "image": cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY),
            "operation": "COLOR_BGRA2GRAY",
            "params": "",
            "show": False,
        }
        return_data["hsv"] = {
            "image": cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV),
            "operation": "COLOR_BGR2HSV",
            "params": "",
            "show": False,
        }
        (_, _, intensity) = cv2.split(return_data["hsv"]["image"])
        return_data["intensity"] = {
            "image": intensity,
            "operation": "extract_v_channel",
            "params": "",
            "show": True,
        }
        return_data["blur"] = {
            "image": image.blur(
                return_data["intensity"]["image"], ksize=(3, 3), iterations=5
            ),
            "operation": "blur",
            "params": image.processing_data[-1],
            "show": True,
        }
        return_data["adaptive_gaussian_threshold"] = {
            "image": image.adaptive_threshold(
                image=return_data["blur"]["image"],
                blockSize=15,
                C=3,
            ),
            "operation": "adaptive_gaussian_threshold",
            "params": image.processing_data[-1],
            "show": True,
        }
        return_data["dilation"] = {
            "image": image.dilate(
                image=return_data["adaptive_gaussian_threshold"]["image"],
                kernel=(5, 5),
                iterations=7,
            ),
            "operation": "dilate",
            "params": image.processing_data[-1],
            "show": True,
        }
        return_data["closing"] = {
            "image": image.closing(
                image=return_data["adaptive_gaussian_threshold"]["image"],
                kernel=(5, 5),
                iterations=10,
            ),
            "operation": "closing",
            "params": image.processing_data[-1],
            "show": True,
        }
        return_data["opening"] = {
            "image": image.opening(
                image=return_data["adaptive_gaussian_threshold"]["image"],
                kernel=(5, 5),
                iterations=7,
            ),
            "operation": "opening",
            "params": image.processing_data[-1],
            "show": True,
        }
        return_data["fill"] = {
            "image": fill(return_data["closing"]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }
        return_data["fill2"] = {
            "image": fill(return_data["opening"]["image"].copy()),
            "operation": "fill2",
            "params": "",
            "show": True,
        }
        return_data["fill3"] = {
            "image": fillCirc(return_data["fill2"]["image"].copy()),
            "operation": "fill3",
            "params": "",
            "show": True,
        }

        return_data["blur_on_fill"] = {
            "image": image.blur(
                return_data["fill"]["image"], ksize=(3, 3), iterations=5
            ),
            "operation": "blur",
            "params": image.processing_data[-1],
            "show": True,
        }

        return return_data

        # return_data["grayscale"] = cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY)
        # return_data["hsv"] = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        # (return_data["hue"], return_data["saturation"], return_data["intensity"]) = cv2.split(
        #     return_data["hsv"]
        # )
        # return_data["blur"] = image.blur(return_data["intensity"], ksize=(3, 3), iterations=5)
        # return_data["adaptive_gaussian_threshold"] = image.adaptive_threshold(
        #     image=return_data["blur"],
        #     blockSize=15,
        #     C=3,
        # )
        # return_data["dilation"] = image.dilate(
        #     return_data["adaptive_gaussian_threshold"],
        #     kernel=(5, 5),
        #     iterations=7,
        # )
        # return_data["closing"] = image.closing(
        #     return_data["adaptive_gaussian_threshold"],
        #     kernel=(5, 5),
        #     iterations=10,
        # )
        # return_data["opening"] = image.opening(
        #     return_data["adaptive_gaussian_threshold"],
        #     kernel=(5, 5),
        #     iterations=7,
        # )
        # return_data["fill"] = fill(return_data["closing"].copy())
        # return_data["fill2"] = fill(return_data["opening"].copy())
        # return_data["fill3"] = fillCirc(return_data["fill2"].copy())

        # image.grayscale = cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY)
        # image.hsv = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        # (image.hue, image.saturation, image.intensity) = cv2.split(image.hsv)

        # image.blur = image.blur(image.intensity, ksize=(3, 3), iterations=5)
        # image.adaptive_gaussian_threshold = image.adaptive_threshold(
        #     image=image.blur,
        #     blockSize=15,
        #     C=3,
        # )
        # image.dilation = image.dilate(
        #     image.adaptive_gaussian_threshold,
        #     kernel=(5, 5),
        #     iterations=7,
        # )
        # image.closing = image.closing(
        #     image.adaptive_gaussian_threshold,
        #     kernel=(5, 5),
        #     iterations=10,
        # )
        # image.opening = image.opening(
        #     image.adaptive_gaussian_threshold,
        #     kernel=(5, 5),
        #     iterations=7,
        # )
        # # image.dialte_opening(image.dilation, kernel=(5, 5), iterations=7)
        # # image.dialte_closing(image.dilation, kernel=(5, 5), iterations=7)
        # image.fill = fill(image.closing.copy())
        # image.fill2 = fill(image.opening.copy())
        # image.fill3 = fillCirc(image.fill2.copy())
