from functools import reduce

import cv2
import numpy as np

from preprocessing import imshow
from similarity import init_background_subtractor


def shift_image(image: np.ndarray, shift: str, val: int = 1):
    """
    Shifts image for val pixels in chosen direction
    :param image: input image matrix
    :param shift: chosen direction
    :param val: number of pixels to shift
    :return: shifted image with the same size as input
    """
    im_copy = image.copy()
    if shift == "top":
        im_copy[:-val, :] = im_copy[val:, :]
    elif shift == "right":
        im_copy[:, val:] = im_copy[:, :-val]
    elif shift == "bottom":
        im_copy[val:, :] = im_copy[:-val, :]
    elif shift == "left":
        im_copy[:, :-val] = im_copy[:, val:]
    return im_copy


if __name__ == "__main__":
    lena = cv2.imread("./data/Lenna.png")
    bgsub = init_background_subtractor(history=1, thresh=50)

    lena_right = shift_image(lena, "right", 5)
    lena_left = shift_image(lena, "left", 5)
    lena_top = shift_image(lena, "top", 5)
    lena_bottom = shift_image(lena, "bottom", 5)

    masks = []
    for lena_shift in [lena_left, lena_top, lena_right, lena_bottom]:
        masks.append(bgsub(lena_shift))
        bgsub(lena)

    mask = reduce(np.logical_and, masks)
    imshow("contours bgsub", mask.astype(np.uint8) * 255)
    cv2.imwrite("./data/results/Lenna_bgsub.png", mask.astype(np.uint8) * 255)

    lena_canny = cv2.Canny(lena, threshold1=30, threshold2=200)
    imshow("contours canny", lena_canny.astype(np.uint8))
    cv2.imwrite("./data/results/Lenna_canny.png", lena_canny.astype(np.uint8))

    imshow("contours bgsub & canny", np.logical_and(lena_canny.astype(np.uint8), mask.astype(np.uint8) * 255).astype(np.uint8) * 255)
    cv2.imwrite("./data/results/Lenna_bgsub_and_canny.png", np.logical_and(lena_canny.astype(np.uint8),
                                                                 mask.astype(np.uint8) * 255).astype(np.uint8) * 255)

    result = cv2.erode(cv2.erode(
               np.logical_and(lena_canny.astype(np.uint8), mask.astype(np.uint8) * 255).astype(np.uint8) * 255,
               np.ones((2, 1), dtype=np.uint8)
           ), np.ones((1, 1), dtype=np.uint8))
    print(result.sum())
    imshow("contours bgsub & canny after erode", result)

    cv2.imwrite("./data/results/Lenna_final.png", result)
