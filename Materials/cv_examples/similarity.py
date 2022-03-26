from typing import Union, Tuple

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from Materials.cv_examples.preprocessing import imshow


def init_background_subtractor(history: int, thresh: int = None, shadows: bool = False, apply_clahe: bool = False):
    """
    Creates a function that when called performs background subtraction with optional histogram equalization
    :param history: number of past frames included in foreground extraction
    :param thresh:
    :param shadows:
    :param apply_clahe:
    :return:
    """
    bg_sub = cv2.createBackgroundSubtractorMOG2(history, thresh, detectShadows=shadows)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    def get_diff(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if apply_clahe:
            frame = clahe.apply(frame)
        frame = bg_sub.apply(frame)
        return frame

    return get_diff


def frame_to_frame_simple_similarity(f1: np.ndarray, f2: np.ndarray):
    # Simple mean and std colour calculation for two frames
    colours1 = f1.reshape((-1, 3))
    colours2 = f2.reshape((-1, 3))
    mean1 = colours1.mean(axis=0)
    mean2 = colours2.mean(axis=0)
    std = (colours1 - colours2).std(axis=0)
    return np.abs(mean1 - mean2), std


def analyze_sift_difference(frame1, frame2, rects_or_contours, display=False):
    """
    Checks if most important SIFT keypoints on differing image pieces are similar or actually represent different shapes
    :param frame1: frame from first video
    :param frame2: frame from second video
    :param rects_or_contours: list of difference contours or rectangles in form of (x, y, w, h)
    :param display: if True, displays debug visuals
    :return:
    """
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher_create(cv2.NORM_L2SQR, crossCheck=True)

    is_same = True
    for region in rects_or_contours:
        if not is_same:
            break

        if type(region) == np.ndarray:
            x, y, w, h = cv2.boundingRect(region)
        else:
            x, y, w, h = region

        kp1, des1 = sift.detectAndCompute(frame1[y:y + h, x:x + w], None)
        kp2, des2 = sift.detectAndCompute(frame2[y:y + h, x:x + w], None)
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            distances = [m.distance for m in matches]

            dist_th = 2 * np.sqrt(w * h) / 3
            cum_distance = 0
            num_kp = 0
            limit_by_distance = int(
                round(max(min(np.sum(np.array(distances) < np.sqrt(distances[-1])), len(distances) / 3), 1)))
            for m in matches:
                i1 = m.queryIdx
                i2 = m.trainIdx
                p1, p2 = kp1[i1], kp2[i2]
                p1, p2 = np.array(p1.pt), np.array(p2.pt)
                dist = np.linalg.norm(p1 - p2)
                if dist > dist_th:
                    is_same = False
                    break
                cum_distance += dist
                num_kp += 1
                if num_kp == limit_by_distance:
                    break
            if not cum_distance < num_kp * (dist_th / 3):
                is_same = False

            if display:
                matches_drawn = cv2.drawMatches(frame1[y:y+h, x:x+w], kp1, frame2[y:y+h, x:x+w], kp2,
                                                matches[:limit_by_distance], None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                imshow("matches!", matches_drawn)
        else:
            # If keypoints are found in the area in one of the images and not found in another's - consider different
            is_same = is_same and (des1 is None and des2 is None or des1 is not None and des2 is not None)

    return not is_same


def get_ssim(im1, im2, rects=None, th=0.6, colour_threshold=30, mse_thresh=0.4, display=False):
    """
    Compute SSIM for pairs of frames regions and deicde based on threshold whether frames are different
    :param im1: first frame
    :param im2: second frame
    :param rects: list of regions in form of (x, y, w, h)
    :param th: deciding threshold for mean ssim (strong outliers threshold is 0.7 * th)
    :param colour_threshold: deciding threshold for mean colour difference
    :param mse_thresh: deciding threshold for mean squared error
    :param display: if True, displays debug visuals
    :return: a tuple - True if SSIM + additional SIFT checks yield difference, False otherwise; and SSIM values list
    """
    if rects is not None:
        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        ssim_list = []
        too_much_colour_diff = False
        for x, y, w, h in rects:
            rect1 = gray1[y:y + h, x:x + w]
            rect2 = gray2[y:y + h, x:x + w]
            ret, val = get_ssim(rect1, rect2)
            # print(ret, val)

            blur1 = cv2.GaussianBlur(im1[y:y + h, x:x + w], (11, 11), 1.5)
            blur2 = cv2.GaussianBlur(im2[y:y + h, x:x + w], (11, 11), 1.5)
            mean_colour_diff, std_colour_diff = frame_to_frame_simple_similarity(blur1, blur2)

            sqrt_proportion = np.sqrt(w * h / im1.shape[0] / im1.shape[1])
            too_much_colour_diff = too_much_colour_diff or any(sqrt_proportion * mean_colour_diff > colour_threshold)

            mse2 = pow((np.abs(blur1.astype(int) - blur2.astype(int)) ** 2).sum(), 1 / 2) / w / h
            too_much_colour_diff = too_much_colour_diff or mse2 > mse_thresh

            if too_much_colour_diff:
                break

            sift_different = False
            if ret:
                # Make sure that we label really different
                sift_different = analyze_sift_difference(rect1, rect2, [[0, 0, w, h]])
                if sift_different or mse2 > 0.3 * mse_thresh:
                    ssim_list.append(val)
            else:
                ssim_list.append(val)

            if display and ret and sift_different:
                imshow(f"ssim result: {ssim_list[-1]}", np.hstack((rect1, rect2)))
                print(f"mse: {np.sqrt(((rect1.astype(np.int16) - rect2) ** 2).mean())}, sift: {sift_different}")
                cv2.destroyWindow(f"ssim result: {ssim_list[-1]}")
        ssim_list = np.array(ssim_list)

        # Radical difference check
        big_th = any(ssim_list < 0.7 * th)
        # Mean deviation check
        if len(ssim_list) > 0:
            mean_th = ssim_list.mean() < th
        else:
            mean_th = False
        # print(too_much_colour_diff, big_th, mean_th, ssim_list)
        return too_much_colour_diff or big_th or mean_th, ssim_list
    ssim_val = ssim(im1, im2, multichannel=False, data_range=255)
    return ssim_val < th, ssim_val