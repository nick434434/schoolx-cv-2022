import time
from asyncio import get_event_loop

import cv2
from base64 import b64encode
from asyncio_mqtt import Client

from Materials.config import CONFIG
from Materials.Lecture_3.camera import VideoProcessor
from Materials.Lecture_4.face_worker import FaceWorker


async def send(image):
    async with Client(CONFIG["mqtt_hostname"]) as client:
        message = b64encode(cv2.imencode(".jpg", image)[1])
        await client.publish("cameras/0", payload=message)


def test():
    image = cv2.imread("data/1.jpg")
    loop = get_event_loop()
    loop.run_until_complete(send(image))


if __name__ == "__main__":
    # test()
    vp = VideoProcessor(0)
    fw = FaceWorker()
    loop = get_event_loop()

    for frame in vp:
        fw.image = frame
        if fw.detect() and len(fw.faces):
            loop.run_until_complete(send(frame))
        time.sleep(0.2)
