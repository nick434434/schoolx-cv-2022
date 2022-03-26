from typing import Union, Tuple

import numpy as np
import cv2


def convolve(image: np.ndarray, kernel: np.ndarray, preset: str = None):
    """
    Compute 2D convolution of an image with the user-provided kernel
    :param image: input image matrix
    :param kernel: convolutional 2D kernel describing the transformation
    :param preset: [optional] if this is set, one of default modes will be used. Kernel parameters serves as a size
    :return: resulting image which is mathe,atically described as [image * kernel]
    """
    if preset is not None:
        kernel = np.ones(tuple(kernel), dtype=np.float32)
    if preset == "median":
        kernel /= kernel.flatten().shape[0]
    elif preset == "gaussian":
       pass

    return cv2.filter2D(image, -1, kernel)


def median(image: np.ndarray, kernel_size: Union[int, Tuple[int, int]]):
    return cv2.medianBlur(image, kernel_size)


def gaussian(image: np.ndarray, kernel_size: Union[int, Tuple[int, int]], sigma: float = None):
    if sigma is None:
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sigma = (ks - 1) / 3
    return cv2.GaussianBlur(image, kernel_size, sigma)


def shadow_removal(image: np.ndarray):
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result, result_norm


def imshow(name: str, frame: np.ndarray, delay: int = 0):
    # Auxiliary function to make displaying debug visualizations faster and simpler
    if frame.shape[0] > 1000:
        frame = cv2.resize(frame, (int(round(1. * frame.shape[1] / frame.shape[0] * 1000)), 1000))
    cv2.imshow(name, frame)
    cv2.waitKey(delay)


if __name__ == "__main__":
    i1 = cv2.imread("./data/Lenna.png")
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

