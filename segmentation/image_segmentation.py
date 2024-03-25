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

    def dilate(self, image, kernel=(5, 5), iterations=3):
        """apply dilation to the image"""
        image = image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
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

    def generic_filter(self, image, kernel, iterations=1, custom_msg="genertic_filter"):
        result = image.copy()

        for i in range(iterations):
            result = cv2.filter2D(result, -1, kernel)

        self.log_image_processing(
            result, f"{custom_msg},kernel:{kernel},iterations:{iterations}"
        )
        return result

    @staticmethod
    def preprocessing2(image):
        image_data = {}

        image_data["original"] = {
            "image": image.image,
            "operation": "original",
            "params": "",
            "show": False,
        }
        image_data["grayscale"] = {
            "image": cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY),
            "operation": "COLOR_BGRA2GRAY",
            "params": "",
            "show": False,
        }
        image_data["hsv"] = {
            "image": cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV),
            "operation": "COLOR_BGR2HSV",
            "params": "",
            "show": False,
        }
        (_, _, intensity) = cv2.split(image_data["hsv"]["image"])
        image_data["intensity"] = {
            "image": intensity,
            "operation": "extract_v_channel",
            "params": "",
            "show": True,
        }
        image_data["blur"] = {
            "image": image.blur(
                image_data["intensity"]["image"], ksize=(9, 9), iterations=2
            ),
            "operation": "blur",
            "params": image.processing_data[-1],
            "show": True,
        }
        # image_data["blur_reverse"] = {
        #     "image": image.blur(
        #         image_data["intensity"]["image"], ksize=(5, 5), iterations=1
        #     ),
        #     "operation": "blur",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["blur_reverse_2_factor_0.25"] = {
        #     "image": image_data["blur"]["image"]
        #     - image_data["blur_reverse"]["image"] * (1 / 4),
        #     "operation": "blur",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        #
        # kernel_sharp = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]])/9
        # image_data["sharpen"] = {
        #     "image": image.generic_filter(
        #         image=image_data["blur"]["image"],
        #         kernel=kernel_sharp,
        #         iterations=1,
        #         custom_msg="sharpen",
        #     ),
        #     "operation": "sharpen",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }

        image_data["adaptive_gaussian_threshold"] = {
            "image": image.adaptive_threshold(
                image=image_data["blur"]["image"].copy(),
                blockSize=15,
                C=2,
            ),
            "operation": "adaptive_gaussian_threshold",
            "params": image.processing_data[-1],
            "show": True,
        }
        image_data["fill_adaptive_gaussian_threshold"] = {
            "image": fill(image_data["adaptive_gaussian_threshold"]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }
        image_data["dilation"] = {
            "image": image.dilate(
                image=image_data["adaptive_gaussian_threshold"]["image"].copy(),
                kernel=(3, 3),
                iterations=1,
            ),
            "operation": "dilate",
            "params": image.processing_data[-1],
            "show": True,
        }
        image_data["fill_dilation"] = {
            "image": fill(image_data["dilation"]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }
        image_data["closing"] = {
            "image": image.closing(
                image=image_data["adaptive_gaussian_threshold"]["image"].copy(),
                kernel=(3, 3),
                iterations=1,
            ),
            "operation": "closing",
            "params": image.processing_data[-1],
            "show": True,
        }
        image_data["fill_closing"] = {
            "image": fill(image_data["closing"]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }
        image_data["opening"] = {
            "image": image.opening(
                image=image_data["adaptive_gaussian_threshold"]["image"].copy(),
                kernel=(3, 3),
                iterations=1,
            ),
            "operation": "opening",
            "params": image.processing_data[-1],
            "show": True,
        }
        image_data["fill_opening"] = {
            "image": fill(image_data["opening"]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }
        image_data["open_on_dilate"] = {
            "image": image.opening(
                image=image_data["dilation"]["image"],
                kernel=(3, 3),
                iterations=1,
            ),
            "operation": "opening",
            "params": image.processing_data[-1],
            "show": True,
        }
        image_data["fill_open_on_dilate"] = {
            "image": fill(image_data["open_on_dilate"]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }

        # image_data["fill3"] = {
        #     "image": fillCirc(image_data["fill2"]["image"].copy()),
        #     "operation": "fill3",
        #     "params": "",
        #     "show": True,
        # }

        # image_data["blur_on_fill"] = {
        #     "image": image.blur(
        #         image_data["fill"]["image"], ksize=(3, 3), iterations=5
        #     ),
        #     "operation": "blur",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }

        return image_data

        # image_data["grayscale"] = cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY)
        # image_data["hsv"] = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        # (image_data["hue"], return_data["saturation"], return_data["intensity"]) = cv2.split(
        #     image_data["hsv"]
        # )
        # image_data["blur"] = image.blur(return_data["intensity"], ksize=(3, 3), iterations=5)
        # image_data["adaptive_gaussian_threshold"] = image.adaptive_threshold(
        #     image=image_data["blur"],
        #     blockSize=15,
        #     C=3,
        # )
        # image_data["dilation"] = image.dilate(
        #     image_data["adaptive_gaussian_threshold"],
        #     kernel=(5, 5),
        #     iterations=7,
        # )
        # image_data["closing"] = image.closing(
        #     image_data["adaptive_gaussian_threshold"],
        #     kernel=(5, 5),
        #     iterations=10,
        # )
        # image_data["opening"] = image.opening(
        #     image_data["adaptive_gaussian_threshold"],
        #     kernel=(5, 5),
        #     iterations=7,
        # )
        # image_data["fill"] = fill(return_data["closing"].copy())
        # image_data["fill2"] = fill(return_data["opening"].copy())
        # image_data["fill3"] = fillCirc(return_data["fill2"].copy())

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
