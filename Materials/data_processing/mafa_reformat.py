from typing import Tuple
import os

import numpy as np
import scipy.io
from cv2 import imread


FACE_WITH_MASK_CLASS = 0
PATH = "/home/anduser/Study/MAFA"
labels_train_path = "MAFA-Label-Train/LabelTrainAll.mat"
labels_test_path = "MAFA-Label-Test/LabelTestAll.mat"
images_train_path = "train-images/images"
images_test_path = "test-images/images"
labels_train_save_folder = "MAFA-yolo-labels/train"
labels_test_save_folder = "MAFA-yolo-labels/test"


def read_mat_file(folder, filepath):
    filename = os.path.join(folder, filepath)
    return scipy.io.loadmat(filename)


def mat_to_files(mat, is_test: bool):
    """

    :param mat:
    :param is_test:
    :return:
    """

    for row in mat:
        im_path = row['imgName'][0][0]
        coco_row = datarow_to_coco(row['label'][0][0], is_test, image_shape=
            imread(os.path.join(PATH, images_test_path if is_test else images_train_path, im_path)).shape[0:2])

        with open(os.path.join(PATH, labels_test_save_folder if is_test else labels_train_save_folder, im_path.split('.')[0]), "a") as f:
            f.writelines([coco_row])


def datarow_to_coco(line, is_test: bool, image_shape: Tuple[int, int]):
    """

    :param line:
    :param is_test:
    :return:
    """
    x, y, w, h = line[:4]

    # if is_test:
    #     face_type = line[1]
    # else:
    #     eye1, eye2 = line[1]
    #
    # occluder = line[1] if is_test else line[2]
    # occluder[0] += x
    # occluder[1] += y
    #
    # occ_type = line[3]
    #
    # occ_degree = line[4]
    #
    # gender_race = line[5]
    #
    # orientation = line[6]
    #
    # glasses = line[7]
    # glasses[0] += x
    # glasses[1] += y

    return f"{FACE_WITH_MASK_CLASS} {x / image_shape[1]} {y / image_shape[0]} {w / image_shape[1]} {h / image_shape[0]}"


if __name__ == "__main__":
    train_mat = read_mat_file(PATH, labels_train_path)['label_train'].transpose((1, 0))
    test_mat = read_mat_file(PATH, labels_test_path)['LabelTest'].transpose((1, 0))

    # os.makedirs(os.path.join(PATH, labels_train_save_folder))
    # os.makedirs(os.path.join(PATH, labels_test_save_folder))

    mat_to_files(train_mat, is_test=False)
    mat_to_files(test_mat, is_test=True)
