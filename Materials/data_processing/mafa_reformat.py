from typing import Tuple
import os
import shutil

import numpy as np
import scipy.io
from cv2 import imread


FACE_WITH_MASK_CLASS = 0
PATH = "/home/anduser/Study/yolov5/datasets/MAFA"
labels_train_path = "MAFA-Label-Train/LabelTrainAll.mat"
labels_test_path = "MAFA-Label-Test/LabelTestAll.mat"
images_train_path = "images/trainMAFA"
images_test_path = "images/valMAFA"
labels_train_save_folder = "labels/trainMAFA"
labels_test_save_folder = "labels/valMAFA"


def read_mat_file(folder, filepath):
    filename = os.path.join(folder, filepath)
    return scipy.io.loadmat(filename)


def mafa_mat_to_coco_files(mat: dict, is_test: bool) -> None:
    """
    Convert every line in MAFA data mat to the YOLO training format (text files with normalised coordinates of the face)
    :param mat: input MATLAB object of certain structure (enable debug mode to inspect closer)
    :param is_test: set to True if the test data is given, False if train
    """

    shutil.rmtree(os.path.join(PATH, labels_test_save_folder if is_test else labels_train_save_folder),
                  ignore_errors=True)
    os.makedirs(os.path.join(PATH, labels_test_save_folder if is_test else labels_train_save_folder))

    for row in mat:
        im_path = row['name'][0][0] if is_test else row['imgName'][0][0]
        coco_row = mafa_datarow_to_coco(row['label'][0][0], is_test, image_shape=
            imread(os.path.join(PATH, images_test_path if is_test else images_train_path, im_path)).shape[0:2])

        with open(os.path.join(PATH, labels_test_save_folder if is_test else labels_train_save_folder, im_path.split('.')[0] + '.txt'), "a") as f:
            f.writelines([coco_row])


def mafa_datarow_to_coco(line: np.ndarray, is_test: bool, image_shape: Tuple[int, int]) -> str:
    """
    Convert a single row of MAFA mat data to YOLO suitable line for one face
    :param line: one MAFA datarow
    :param is_test: set to True if the test data is given, False if train
    :param image_shape: shape of the image for which the data is being transformed
    :return: YOLO-formatted string for respective MAFA datarow
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

    return f"{FACE_WITH_MASK_CLASS} {x / image_shape[1] + w / image_shape[1] / 2} " \
           f"{y / image_shape[0] + h / image_shape[0] / 2} {w / image_shape[1]} {h / image_shape[0]}"


def mafa_create_coco_listings(mat: dict, is_test: bool):
    """

    :param mat: input MATLAB object of certain structure (enable debug mode to inspect closer)
    :param is_test: set to True if the test data is given, False if train
    :return:
    """
    with open(os.path.join(PATH, "valMAFA.txt" if is_test else "trainMAFA.txt"), "w") as f:
        for row in mat:
            im_path = row['name'][0][0] if is_test else row['imgName'][0][0]
            label_file_name = \
              os.path.join('.', images_test_path if is_test else images_train_path, im_path)
            f.write(label_file_name + '\n')


if __name__ == "__main__":
    train_mat = read_mat_file(PATH, labels_train_path)['label_train'].transpose((1, 0))
    test_mat = read_mat_file(PATH, labels_test_path)['LabelTest'].transpose((1, 0))

    # os.makedirs(os.path.join(PATH, labels_train_save_folder))
    # os.makedirs(os.path.join(PATH, labels_test_save_folder))

    mafa_mat_to_coco_files(train_mat, is_test=False)
    mafa_mat_to_coco_files(test_mat, is_test=True)

    # mafa_create_coco_listings(train_mat, False)
    # mafa_create_coco_listings(test_mat, True)

    # for filename in os.listdir(os.path.join(PATH, labels_train_save_folder)):
    #     full_filename = os.path.join(PATH, labels_train_save_folder, filename)
    #     if not full_filename.endswith('.txt'):
    #         os.remove(full_filename + '.txt')
    #         os.rename(full_filename, full_filename + '.txt')
    #
    # for filename in os.listdir(os.path.join(PATH, labels_test_save_folder)):
    #     full_filename = os.path.join(PATH, labels_test_save_folder, filename)
    #     if not full_filename.endswith('.txt'):
    #         os.remove(full_filename + '.txt')
    #         os.rename(full_filename, full_filename + '.txt')
