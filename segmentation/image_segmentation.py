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
        if type(op) != np.ndarray:
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
        if type(op) != np.ndarray:
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

    def closing(self, image, kernel=(5, 5), iterations=10, op=cv2.MORPH_ELLIPSE):
        """apply closing to the image"""
        image = image.copy()
        if type(op) != np.ndarray:
            kernel = cv2.getStructuringElement(op, kernel)
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
        if type(op) != np.ndarray:
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

    def fill_image(self, image_data, name, show=False):
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

    def find_ball_contours(
        self, image, circ_thresh, alpha=1, min_area=400, max_area=4900, verbose=False
    ):
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
            circularity = 4 * math.pi * area / (peri**2)

            # circularity_constant = (
            #     1 / (1 - circ_thresh / (max_area)) * (area + min_area) + circ_thresh
            # )
            if len(approx) < 5 or area < min_area or area > max_area:
                continue

            if circularity < circ_thresh:
                # circ_thresh_2 = circ_thresh - 0.15
                circularity_constant = (
                    -(1 - circ_thresh) / (max_area - min_area) * (area) + circ_thresh
                ) / alpha

                if circularity < circ_thresh - circularity_constant:
                    # print(
                    #     f"rejected area {area}, circularity_constant {circularity_constant}, circularity {circularity}"
                    # )
                    continue

                # print(
                #     f"area {area}, circularity_constant {circularity_constant}, circularity {circularity}"
                # )
                #
                # if verbose:
                # print("\ndetected")
                # print("approx", len(approx))
                # print("area", area)
                # print("peri", peri)
                # if circularity < circ_thresh + 0.1:
                # print(
                #     f"circ: {circularity}, area: {area}, circularity_constant: {circularity_constant}"
                # )
                # if len(approx) < circ + 10 and len(approx) > circ - 10:
            cv2.drawContours(blank_image, [c], -1, (255), cv2.FILLED)
            # cv2.ellipse(blank_image, c, (255), 2)
        # cv2.imshow("cnts", blank_image)
        # cv2.waitKey(0)

        # print(blank_image.max())

        return blank_image

    @staticmethod
    def preprocessing(image):
        image_data = {}

        image_data["original"] = {
            "image": image.image,
            "show": False,
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
            image_data["intensity"]["image"], 125, 255, cv2.THRESH_BINARY
        )[1]

        image_data["intensity_threshold"] = {
            "image": intensity_threshold,
            "show": False,
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
                k_d=4,
                i_d=5,
                k_e=5,
                i_e=2,
                iterations=1,
                # op=cv2.MORPH_CROSS,
            ),
            "show": False,
        }

        image_data["segmentation_with_threshold"] = {
            "image": cv2.bitwise_not(
                cv2.bitwise_or(
                    image_data["intensity_threshold"]["image"],
                    image_data["dilate_and_erode"]["image"],
                ),
            ),
            "show": False,
        }

        contours = image.find_ball_contours(
            cv2.bitwise_not(image_data["dilate_and_erode"]["image"]),
            circ_thresh=0.3,
            alpha=1,
            min_area=400,
            max_area=4900,
            verbose=False,
        )

        image_data["contours"] = {
            "image": contours,
            "show": False,
        }

        image_data["im_1"] = {
            "image": cv2.bitwise_not(
                image_data["intensity_threshold"]["image"],
            ),
            "show": False,
        }

        image_data["im_2"] = {
            "image": cv2.bitwise_not(
                image_data["contours"]["image"],
            ),
            "show": False,
        }

        image_data["segmentation_before_recontour"] = {
            "image": cv2.bitwise_not(
                cv2.bitwise_or(
                    image_data["im_1"]["image"], image_data["im_2"]["image"]
                ),
            ),
            "show": False,
        }
        # image_data["dilate_after_opening_after_segmentation"] = {
        #     "image": image.dilate(
        #         image_data["segmentation_before_recontour"]["image"],
        #         kernel=(4, 4),
        #         iterations=3,
        #     ),
        #     "show": True,
        # }
        # image_data["opening_after_segmentation"] = {
        #     "image": image.opening(
        #         image_data["dilate_after_opening_after_segmentation"]["image"],
        #         kernel=(5, 5),
        #         iterations=5,
        #     ),
        #     "show": True,
        # }
        # image_data["erode_opening_after_segmentation"] = {
        #     "image": image.erode(
        #         image_data["opening_after_segmentation"]["image"],
        #         kernel=(3, 3),
        #         iterations=4,
        #     ),
        #     "show": True,
        # }
        #
        circular_se = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype="uint8",
        )
        image_data["erode_after_segmentation"] = {
            "image": image.erode(
                image_data["segmentation_before_recontour"]["image"],
                kernel=(3, 3),
                iterations=2,
                # op=cv2.MORPH_RECT,
            ),
            "show": False,
        }

        image_data["dilate_after_erode_after_segmentation"] = {
            "image": image.dilate(
                image_data["erode_after_segmentation"]["image"],
                kernel=(2, 2),
                iterations=8,
                op=circular_se,
            ),
            "show": False,
        }

        # image_data["closing_after_dilate_after_erode_after_segmentation"] = {
        #     "image": image.closing(
        #         image_data["dilate_after_erode_after_segmentation"]["image"],
        #         kernel=(7, 7),
        #         iterations=4,
        #         op=circular_se,
        #     ),
        #     "show": True,
        # }

        image_data["opening_after_dilate_after_erode_after_segmentation"] = {
            "image": image.opening(
                image_data["dilate_after_erode_after_segmentation"]["image"],
                kernel=(7, 7),
                iterations=3,
                op=circular_se,
            ),
            "show": False,
        }

        image_data[
            "erode_after_opening_after_dilate_after_erode_after_segmentation"
        ] = {
            "image": image.erode(
                image_data["opening_after_dilate_after_erode_after_segmentation"][
                    "image"
                ],
                kernel=(3, 3),
                iterations=7,
                # op=circular_se,
            ),
            "show": True,
        }

        circular_se = np.array(
            [
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
            ]
        )

        image_data[
            "dilate_after_erode_after_opening_after_dilate_after_erode_after_segmentation"
        ] = {
            "image": image.dilate(
                image_data[
                    "erode_after_opening_after_dilate_after_erode_after_segmentation"
                ]["image"],
                kernel=(3, 3),
                iterations=5,
                # op=circular_se,
            ),
            "show": True,
        }

        # image_data["opening_after_segmentation_2"] = {
        #     "image": image.opening(
        #         image_data["dilate_after_opening_after_segmentation"]["image"],
        #         kernel=(2, 2),
        #         iterations=10,
        #     ),
        #     "show": True,
        # }

        thresh = 1
        min_area = 150
        for i in range(1, 3):
            i = i
            image_data[f"segmentation_thresh_{thresh}_alpha_{i}"] = {
                "image": image.find_ball_contours(
                    image_data[
                        "dilate_after_erode_after_opening_after_dilate_after_erode_after_segmentation"
                    ]["image"],
                    circ_thresh=thresh,  # range betwen .65 and .7
                    alpha=i,
                    min_area=min_area,
                    max_area=3500,
                    verbose=True,
                ),
                "show": True,
            }

        image_data[f"segmentation"] = {
            "image": image.dilate(
                image_data[f"segmentation_thresh_{thresh}_alpha_1"]["image"],
                kernel=(3, 3),
                iterations=3,
                # op=circular_se,
            ),
            "show": True,
        }

        # image_data["segmentation"] = {
        #     "image": image.find_ball_contours(
        #         image_data[
        #             "dilate_after_erode_after_opening_after_dilate_after_erode_after_segmentation"
        #         ]["image"],
        #         circ_thresh=thresh,  # range betwen .65 and .7
        #         alpha=3,
        #         min_area=min_area,
        #         max_area=3500,
        #         verbose=True,
        #     ),
        #     "show": False,
        # }

        return image_data
