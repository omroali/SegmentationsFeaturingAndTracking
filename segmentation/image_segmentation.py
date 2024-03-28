import os
import cv2
from cv2.typing import MatLike
import numpy as np
from segmentation.utils import fill
import math


# @dataclass(order=True)
# class ImageData:
#     image: MatLike
#     operation: str = ""
#     params: str = ""
#     show: bool = True


class ImageSegmentation:
    def __init__(self, image_path: str, save_dir: str = None):
        self.processing_data = []
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.processing_images = []
        self.save_dir = save_dir

    def log_image_processing(self, image, operation: str):
        """log the image processing"""
        self.processing_data.append(operation)
        self.processing_images.append(image)

    def gblur(self, image, ksize=(3, 3), iterations=1):
        """apply gaussian blur to the image"""
        blur = image.copy()
        for _ in range(iterations):
            blur = cv2.GaussianBlur(blur, ksize, cv2.BORDER_DEFAULT)
        self.log_image_processing(blur, f"gblur,kernel:{ksize},iterations:{iterations}")
        return blur

    def mblur(self, image, ksize=3, iterations=1):
        """apply gaussian blur to the image"""
        blur = image.copy()
        for _ in range(iterations):
            blur = cv2.medianBlur(blur, ksize)
        self.log_image_processing(
            blur, f"medianblur,kernel:{ksize},iterations:{iterations}"
        )
        return blur

    def adaptive_threshold(self, image, blockSize=15, C=3):
        """apply adaptive threshold to the image"""
        image = image.copy()
        adaptive_gaussian_threshold = cv2.adaptiveThreshold(
            src=image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,
            C=C,
        )
        self.log_image_processing(
            adaptive_gaussian_threshold,
            f"adaptive_threshold,blockSize:{blockSize},C:{C}",
        )
        return adaptive_gaussian_threshold

    def dilate(self, image, kernel=(3, 3), iterations=1, op=cv2.MORPH_ELLIPSE):
        """apply dilation to the image"""
        image = image.copy()
        kernel = cv2.getStructuringElement(op, kernel)
        dilation = cv2.dilate(
            src=image,
            kernel=kernel,
            iterations=iterations,
        )

        self.log_image_processing(
            dilation,
            f"erode,kernel:{kernel},iterations:{iterations}",
        )
        return dilation

    def erode(self, image, kernel=(3, 3), iterations=1, op=cv2.MORPH_ELLIPSE):
        """apply dilation to the image"""
        image = image.copy()
        kernel = cv2.getStructuringElement(op, kernel)
        dilation = cv2.erode(
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

    def opening(self, image, kernel=(5, 5), iterations=1, op=cv2.MORPH_ELLIPSE):
        """apply opening to the image"""
        image = image.copy()
        kernel = cv2.getStructuringElement(op, kernel)
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

    def dilate_and_erode(
        self, image, k_d, i_d, k_e, i_e, iterations=1, op=cv2.MORPH_ELLIPSE
    ):
        image = image.copy()
        for _ in range(iterations):
            for _ in range(i_d):
                image = self.dilate(image, (k_d, k_d), op=op)
            for _ in range(i_e):
                image = self.erode(image, (k_e, k_e), op=op)
        self.log_image_processing(
            image,
            f"dilate_and_erode,k_d:{(k_d,k_d)},i_d={i_d},k_e:{(k_e, k_e)},i_e={i_e},iterations:{iterations}",
        )
        return image

    def fill_image(self, image_data, name, show=True):
        self.log_image_processing(
            image_data[name],
            f"fill_{name}",
        )
        image_data[f"fill_{name}"] = {
            "image": fill(image_data[name]["image"].copy()),
            "show": show,
        }

    # FIXME: remove this method
    # def threshold(self, image_data, name, min=0, max=255):
    #     self.log_image_processing(
    #         image_data[name],
    #         f"threshold,min={min},max={max}",
    #     )
    #         image_data[f"fill_{name}"] = {
    #         "image": cv2.threshold(
    #             image_data[name]["image"].copy(),
    #         ),
    #         "operation": "threshold",
    #         "params": f"min={min},max={max}",
    #         "show": True,
    #     }

    # def hoff_transform(self, image):
    #     img = image.copy()
    #     circles = cv2.HoughCircles(
    #         img,
    #         cv2.HOUGH_GRADIENT,
    #         1,
    #         20,
    #         param1=50,
    #         param2=30,
    #         minRadius=10,
    #         maxRadius=100,
    #     )

    #     # Draw detected circles
    #     if circles is not None:
    #         circles = np.uint16(np.around(circles))
    #         for i in circles[0, :]:
    #             cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #             cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    #     # cv2.imshow("Detected Circles", img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     return img

    def find_ball_contours(self, image):
        img = image.copy()
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        blank_image = np.zeros(img.shape, dtype=img.dtype)

        # Loop over contours
        # print("\n\n -- next frame --")
        for c in cnts:
            # Calculate properties
            peri = cv2.arcLength(c, True)

            # Douglas-Peucker algorithm
            approx = cv2.approxPolyDP(c, 0.0001 * peri, True)
            area = cv2.contourArea(c)
            if area == 0:
                continue  # Skip to the next iteration if area is zero
            # Filter for circular shapes
            radius = math.sqrt(area / math.pi)
            circ = int(2 * math.pi * radius)

            if (len(approx) > 10) and (area > 400 and area < 4900):
                # print("\ndetected")
                # print("approx", len(approx))
                # print("area", area)
                # print("peri", peri)
                # print("circ", circ)
                # if len(approx) < circ + 10 and len(approx) > circ - 10:
                cv2.drawContours(blank_image, [c], -1, (255), cv2.FILLED)
                # cv2.ellipse(blank_image, c, (255), 2)
        # cv2.imshow("cnts", blank_image)
        # cv2.waitKey(0)

        # print(blank_image.max())

        return blank_image

    @staticmethod
    def evaluating_effect_of_blocksize(image, C):
        image_data = {}
        image_data["original"] = {
            "image": image.image,
            "operation": "original",
            "show": False,
        }

        (_, _, intensity) = cv2.split(cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV))
        image_data["intensity"] = {
            "image": cv2.threshold(
                image_data["intensity"], 100, 255, cv2.THRESH_BINARY_INV
            ),
            "show": True,
        }
        # image_data["blur"] = {
        #     "image": image.blur(
        #         image_data["intensity"]["image"], ksize=(3, 3), iterations=2
        #     ),
        #     "operation": "blur",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }

        for blockSize in range(5, 21, 2):
            name = f"adap_gaus_thrsh_bS_{blockSize}_C_{C}"
            image_data[name] = {
                "image": image.adaptive_threshold(
                    image=image_data["blur"]["image"].copy(),
                    blockSize=blockSize,
                    C=C,
                ),
                "show": False,
            }

            image.fill_image(image_data, name)

    @staticmethod
    def preprocessing2(image):
        image_data = {}

        image_data["original"] = {
            "image": image.image,
            "show": True,
        }
        image_data["grayscale"] = {
            "image": cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY),
            "show": False,
        }
        image_data["hsv"] = {
            "image": cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV),
            "show": False,
        }
        (_, _, intensity) = cv2.split(image_data["hsv"]["image"])
        image_data["intensity"] = {
            "image": intensity,
            "show": False,
        }
        image_data["gblur"] = {
            "image": image.gblur(
                image_data["intensity"]["image"], ksize=(3, 3), iterations=2
            ),
            "show": False,
        }
        image_data["blur"] = {
            "image": image.mblur(
                image_data["intensity"]["image"], ksize=3, iterations=2
            ),
            "show": False,
        }

        intensity_threshold = cv2.threshold(
            image_data["intensity"]["image"], 100, 255, cv2.THRESH_BINARY
        )[1]

        image_data["intensity_threshold"] = {
            "image": intensity_threshold,
            "show": True,
        }

        name = "adap_gaus_thrsh"
        image_data[name] = {
            "image": image.adaptive_threshold(
                image=image_data["blur"]["image"].copy(),
                blockSize=19,
                C=5,
            ),
            "show": False,
        }

        image.fill_image(image_data, name)

        # contours = image.find_ball_contours(
        #     cv2.bitwise_not(image_data[f"fill_{name}"]["image"])
        # )
        # image_data["contours"] = {
        #     "image": contours,
        #     "operation": "",
        #     "params": "",
        #     "show": True,
        # }

        image_data["open"] = {
            "image": image.opening(
                image=image_data["adap_gaus_thrsh"]["image"].copy(),
                kernel=(5, 5),
                iterations=4,
            ),
            "show": False,
        }

        image_data["erode"] = {
            "image": image.erode(
                image=image_data["open"]["image"].copy(),
                kernel=(3, 3),
                iterations=2,
            ),
            "show": False,
        }
        image.fill_image(image_data, "erode")
        image_data["dilate_and_erode"] = {
            "image": image.dilate_and_erode(
                image_data["fill_erode"]["image"].copy(),
                k_d=5,
                i_d=3,
                k_e=10,
                i_e=1,
                iterations=3,
                op=cv2.MORPH_CROSS,
            ),
            "show": False,
        }
        image_data["dilate_and_erode_2"] = {
            "image": image.dilate_and_erode(
                image_data["fill_erode"]["image"].copy(),
                k_d=6,
                i_d=3,
                k_e=7,
                i_e=2,
                iterations=2,
                # op=cv2.MORPH_CROSS,
            ),
            "show": False,
        }
        image_data["dilate_2"] = {
            "image": image.dilate(
                image=image_data["dilate_and_erode"]["image"].copy(),
                kernel=(4, 4),
                iterations=5,
            ),
            "show": False,
        }
        image_data["dilate_and_erode_3"] = {
            "image": image.dilate_and_erode(
                image_data["fill_erode"]["image"].copy(),
                k_d=4,
                i_d=5,
                k_e=5,
                i_e=2,
                iterations=1,
                # op=cv2.MORPH_CROSS,
            ),
            "show": False,
        }

        segmentation = image.find_ball_contours(
            cv2.bitwise_not(image_data["dilate_and_erode_3"]["image"])
        )

        segmentation_with_threshold = cv2.bitwise_and(
            segmentation,
            image_data["intensity_threshold"]["image"],
        )

        image_data["segmentation"] = {
            "image": segmentation_with_threshold,
            "show": True,
        }

        return image_data
