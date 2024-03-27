import cv2
from cv2.typing import MatLike
import numpy as np
from segmentation.utils import fill
import math



@dataclass(order=True)
class ImageData:
    image: MatLike
    operation: str = ""
    params: str = ""
    show: bool = True


class ImageSegmentation:
    def __init__(self, image_path: str):
        self.processing_data = []
        self.image = cv2.imread(image_path)
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

    def dilate(self, image, kernel=(3, 3), iterations=1):
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
            f"erode,kernel:{kernel},iterations:{iterations}",
        )
        return dilation

    def erode(self, image, kernel=(3, 3), iterations=1):
        """apply dilation to the image"""
        image = image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
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

    def dilateAndErode(self, image, k_d, k_e, iterations=1):
        image = image.copy()
        for i in range(iterations):
            image = self.dilate(image, (k_d, k_d))
            image = self.erode(image, (k_e, k_e))
        self.log_image_processing(
            image,
            f"dilate_and_erode,k_d:{(k_d,k_d)},k_e:{(k_e, k_e)},iterations:{iterations}",
        )
        return image

    def fill_image(self, image_data, name):

        self.log_image_processing(
            image_data[name],
            f"fill_{name}",
        )
        image_data[f"fill_{name}"] = {
            "image": fill(image_data[name]["image"].copy()),
            "operation": "fill",
            "params": "",
            "show": True,
        }

    def threshold(self, image_data, name, min=0, max=255):
        self.log_image_processing(
            image_data[name],
            f"threshold,min={min},max={max}",
        )
        image_data[f"fkjill_{name}"] = {
            "image": cv2.threshold(
                image_data[name]["image"].copy(),
            ),
            "operation": "threshold",
            "params": f"min={min},max={max}",
            "show": True,
        }

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

            if (len(approx) > 10) and (area > 300 and area < 1000):
                if len(approx) < circ + 10 and len(approx) > circ - 10:
                    # print("\ndetected")
                    # print("approx", len(approx))
                    # print("area", area)
                    # print("peri", peri)
                    # print("circ", circ)
                    cv2.drawContours(blank_image, [c], -1, (255), 2)
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
            "params": "",
            "show": False,
        }

        (_, _, intensity) = cv2.split(cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV))
        image_data["intensity"] = {
            "image": cv2.threshold(
                image_data["intensity"], 100, 255, cv2.THRESH_BINARY_INV
            ),
            "operation": "extract_v_channel",
            "params": "",
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
                "operation": "adaptive_gaussian_threshold",
                "params": image.processing_data[-1],
                "show": False,
            }

            image.fill_image(image_data, name)

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
        # image_data["normalize"] = {
        #     "image": cv2.normalize(image_data["intensity"]["image"], cv2.NORM_RELATIVE),
        #     "operation": "blur",
        #     "params": "",
        #     "show": True,
        # }
        image_data["blur"] = {
            "image": image.blur(
                image_data["intensity"]["image"], ksize=(3, 3), iterations=3
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
<<<<<<< Updated upstream
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
        # blockSize = 19
        # C = 4.7
        name = "adap_gaus_thrsh"
        image_data[name] = {
            "image": image.adaptive_threshold(
                image=image_data["blur"]["image"].copy(),
                # blockSize=blockSize,
                # C=C,
            ),
            "operation": "adaptive_gaussian_threshold",
            "params": image.processing_data[-1],
            "show": True,
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
                iterations=3,
            ),
            "operation": "opening",
            "params": image.processing_data[-1],
            "show": True,
        }

        image_data["open"] = {
            "image": image.opening(
                image=image_data["open"]["image"].copy(),
                kernel=(3, 3),
                iterations=1,
            ),
            "operation": "opening",
            "params": image.processing_data[-1],
            "show": True,
        }
        image.fill_image(image_data, "open")

        image_data["close"] = {
            "image": image.closing(
                image=image_data["fill_open"]["image"].copy(),
               kernel=(5, 5),
                iterations=3,
            ),
            "operation": "close",
            "params": image.processing_data[-1],
            "show": True,
        }

        image.fill_image(image_data, "close")

        image_data["dilate"] = {
            "image": image.dilate(
                image=image_data["fill_open"]["image"].copy(),
                kernel=(5, 5),
                iterations=3,
            ),
            "operation": "dilate",
            "params": image.processing_data[-1],
            "show": True,
        }

        image.fill_image(image_data, "dilate")
        #
        # image_data["dilate"] = {
        #     "image": image.dilate(
        #         image=image_data["fill_open"]["image"].copy(),
        #         kernel=(6, 6),
        #         iterations=2,
        #     ),
        #     "operation": "dialte",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        #
        # image.fill_image(image_data, "dilate")

        # image_data["fill_open"] = {
        #     "image": fill(image_data["open"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        #
        # image_data["dilate_and_erode_on_open"] = {
        #     "image": image.dilateAndErode(
        #         image=image_data["open"]["image"].copy(),
        #         k_d=3,
        #         k_e=3,
        #         iterations=7,
        #     ),
        #     "operation": "opening",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["fill_dilate_and_erode_on_open"] = {
        #     "image": fill(image_data["dilate_and_erode_on_open"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",

        # image_data["dilation_on_erode"] = {
        #     "image": image.dilate(
        #         image=image_data["erode"]["image"].copy(),
        #         kernel=(3, 3),
        #         iterations=7,
        #     ),
        #     "operation": "dilate",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["fill_dilation_on_erode"] = {
        #     "image": fill(image_data["dilation_on_erode"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }

        # image_data["erode_on_dilation_on_erode"] = {
        #     "image": image.erode(
        #         image=image_data["dilation_on_erode"]["image"].copy(),
        #         kernel=(5, 5),
        #         iterations=4,
        #     ),
        #     "operation": "dilate",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["fill_erode_on_dilation_on_erode"] = {
        #     "image": fill(image_data["erode_on_dilation_on_erode"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        # image_data["open_on_dilation_on_erode_k3_i9"] = {
        #     "image": image.opening(
        #         image=image_data["dilation_on_erode"]["image"].copy(),
        #         kernel=(3, 3),
        #         iterations=9,
        #     ),
        #     "operation": "opening",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["fill_open_on_dilation_on_erode_k3_i9"] = {
        #     "image": fill(
        #         image_data["open_on_dilation_on_erode_k3_i9"]["image"].copy()
        #     ),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        # image_data["open_on_dilation_on_erode_k5_i9"] = {
        #     "image": image.opening(
        #         image=image_data["dilation_on_erode"]["image"].copy(),
        #         kernel=(5, 5),
        #         iterations=9,
        #     ),
        #     "operation": "opening",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["fill_open_on_dilation_on_erode_k5_i9"] = {
        #     "image": fill(
        #         image_data["open_on_dilation_on_erode_k5_i9"]["image"].copy()
        #     ),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        #
        # image_data["closing_on_dilation_on_erode"] = {
        #     "image": image.closing(
        #         image=image_data["dilation_on_erode"]["image"].copy(),
        #         kernel=(5, 5),
        #         iterations=2,
        #     ),
        #     "operation": "closing",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        # image_data["fill_closing"] = {
        #     "image": fill(image_data["closing_on_dilation_on_erode"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        #
        # image_data["erode_on_dilation_on_erode"] = {
        #     "image": image.erode(
        #         image=image_data["adaptive_gaussian_threshold"]["image"].copy(),
        #         kernel=(3, 3),
        #         iterations=3,
        #     ),
        #     "operation": "erode",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        #
        # image_data["fill_erode_on_dilation_on_erode"] = {
        #     "image": fill(image_data["erode_on_dilation_on_erode"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        #
        # image_data["fill_open_on_dilate"] = {
        #     "image": fill(image_data["open_on_dilate"]["image"].copy()),
        #     "operation": "fill",
        #     "params": "",
        #     "show": True,
        # }
        #
        # image_data["fill3"] = {
        #     "image": fillCirc(image_data["fill2"]["image"].copy()),
        #     "operation": "fill3",
        #     "params": "",
        #     "show": True,
        # }
        #
        # image_data["blur_on_fill"] = {
        #     "image": image.blur(
        #         image_data["fill"]["image"], ksize=(3, 3), iterations=5
        #     ),
        #     "operation": "blur",
        #     "params": image.processing_data[-1],
        #     "show": True,
        # }
        #
        return image_data
        #
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
        #
        # image.grayscale = cv2.cvtColor(image.image, cv2.COLOR_BGRA2GRAY)
        # image.hsv = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        # (image.hue, image.saturation, image.intensity) = cv2.split(image.hsv)
        #
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
