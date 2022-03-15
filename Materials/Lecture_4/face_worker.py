from enum import Enum
from typing import Union, Callable, List

import numpy as np
import dlib
import cv2

from Materials.config import CONFIG


class FaceStages(Enum):
    LOADED_IMAGE = 0
    DETECTED_FACES = 1
    FOUND_LANDMARKS = 2
    EXTRACTED_PATCHES = 3
    EXTRACTED_FEATURES = 4


def dlib_numpy_rect_converter(main_data, convert_to, additional_data=None):
    if not isinstance(main_data, convert_to):
        if convert_to == np.ndarray:
            if isinstance(main_data, dlib.full_object_detection):
                return np.array([(part.x, part.y) for part in main_data.parts()], dtype=np.float32)
            elif isinstance(main_data, dlib.rectangle):
                return np.array((main_data.left(), main_data.top(), main_data.right(), main_data.bottom()), dtype=np.float32)
        elif convert_to == dlib.rectangle:
            return dlib.rectangle(*main_data)
        elif convert_to == dlib.full_object_detection:
            return dlib.full_object_detection(rect=main_data, parts=dlib.points([dlib.point(point) for point in additional_data]))
    else:
        return main_data


def draw_landmarks(image, landmarks):
    draw_image = image.copy()
    for x, y in landmarks:
        draw_image = cv2.circle(draw_image, (int(x), int(y)), 2, (0, 255, 0), 2)
    return draw_image


class FaceWorker:
    __slots__ = [
        "image", "faces", "face_images", "landmarks", "features", "stage",
        "detector", "landmark_finder", "recognizer",
        "detector_type", "landmarks_type", "recognizer_type",
    ]

    def __init__(self, image: np.ndarray = None, detector_type: str = "dlib", landmarks_type: str = "dlib",
                 recognizer_type: str = "dlib"):
        # Data init
        self.image: np.ndarray = image
        self.faces: Union[None, List[Union[dlib.rectangle, np.ndarray]], dlib.rectangles] = None
        self.face_images: Union[None, List[np.ndarray]] = None
        self.landmarks: Union[None, List[np.ndarray]] = None
        self.features: Union[None, List[np.ndarray]] = None

        self.stage: FaceStages = FaceStages.LOADED_IMAGE

        # Models and their settings init
        self.detector_type: str = detector_type
        self.landmarks_type: str = landmarks_type
        self.recognizer_type: str = recognizer_type
        self.detector: Union[None, Callable] = None
        self.landmark_finder: Union[None, Callable] = None
        self.recognizer: Union[None, Callable] = None

        # Actually load models into memory
        self.setup()

    def reset(self):
        self.faces: Union[None, List[Union[dlib.rectangle, np.ndarray]], dlib.rectangles] = None
        self.face_images: Union[None, List[np.ndarray]] = None
        self.landmarks: Union[None, List[np.ndarray]] = None
        self.features: Union[None, List[np.ndarray]] = None
        self.stage: FaceStages = FaceStages.LOADED_IMAGE

    def setup(self):
        if self.detector_type == "dlib":
            self.detector = dlib.get_frontal_face_detector()
        else:
            raise NotImplementedError
        if self.landmarks_type == "dlib":
            self.landmark_finder = dlib.shape_predictor(CONFIG["shape_predictor_path"])
        else:
            raise NotImplementedError
        if self.recognizer_type == "dlib":
            self.recognizer = dlib.face_recognition_model_v1(CONFIG["dlib_recognizer_path"])
        else:
            raise NotImplementedError

    def detect(self):
        self.faces = self.detector(self.image)
        if self.faces is not None:
            self.stage = FaceStages.DETECTED_FACES
            return True
        return False

    def find_landmarks(self):
        if not self.stage == FaceStages.DETECTED_FACES:
            return False
        self.landmarks = []
        for face in self.faces:
            rect = dlib_numpy_rect_converter(face, convert_to=dlib.rectangle)
            self.landmarks.append(dlib_numpy_rect_converter(self.landmark_finder(self.image, rect),
                                                            convert_to=np.ndarray))
        self.stage = FaceStages.FOUND_LANDMARKS
        return True

    def get_normalized_faces(self):
        self.face_images = []
        for i, face in enumerate(self.faces):
            landmarks = self.landmarks[i]
            self.face_images.append(dlib.get_face_chip(self.image,
                                                       dlib_numpy_rect_converter(self.faces[i],
                                                                                 convert_to=dlib.full_object_detection,
                                                                                 additional_data=landmarks)))
        self.stage = FaceStages.EXTRACTED_PATCHES

    def extract_features(self):
        self.features = []
        if self.stage == FaceStages.EXTRACTED_PATCHES:
            for i, face in enumerate(self.faces):
                fod = dlib_numpy_rect_converter(face, dlib.full_object_detection, self.landmarks[i])
                feature = self.recognizer.compute_face_descriptor(self.image, face=fod)
                self.features.append(np.array(feature))
            self.stage = FaceStages.EXTRACTED_FEATURES

    def run(self):
        # Run all the stages from the scratch
        self.reset()
        self.detect()
        self.find_landmarks()
        self.get_normalized_faces()
        self.extract_features()

    def show(self, cap: str = ""):
        for i, face in enumerate(self.faces):
            x0, y0, x1, y1 = map(int, dlib_numpy_rect_converter(face, convert_to=np.ndarray))
            face_im = draw_landmarks(self.image[y0:y1, x0:x1], self.landmarks[i] - np.array((x0, y0), dtype=int))
            cv2.imshow(f"{cap} face {i + 1}", face_im)
            cv2.waitKey(0)

    def __call__(self, image: np.ndarray = None):
        if image is not None:
            self.image = image
        elif self.image is None:
            raise ValueError("if image was not setup before, you need to set it in object call")
        self.run()


if __name__ == "__main__":
    img = cv2.imread("./data/face_image.png")
    fw = FaceWorker(img)
    fw()
    fw.show()
