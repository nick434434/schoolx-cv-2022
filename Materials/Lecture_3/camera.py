from typing import Union
from base64 import b64encode
import traceback
from threading import Thread

import numpy as np
import cv2


class VideoProcessor:
    def __init__(self, video_path: Union[int, str], encode: bool = False):
        self.path = video_path
        self.cap = cv2.VideoCapture(self.path)
        if not (isinstance(video_path, int) or isinstance(video_path, str)):
            raise TypeError
        if not self.cap.isOpened():
            raise ValueError(f"video_path parameter value ({video_path}) does not represent a valid video or stream")
        self.opened = True
        self.frame: Union[None, np.ndarray, str] = None
        self.encode = encode

    def _next(self):
        if self.opened:
            self.opened, frame = self.cap.read()
            if self.opened:
                self.frame = str(b64encode(cv2.imencode(".jpg", frame)[-1]))[2:-1] if self.encode else frame

    def __iter__(self):
        while True:
            self._next()
            self.opened = self.opened and self.cap.isOpened()
            if not self.opened:
                break
            yield self.frame


def declarative_alternative(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("declarative webcam", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    vp = VideoProcessor(0)
    for frame in vp:
        cv2.imshow("webcam", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
