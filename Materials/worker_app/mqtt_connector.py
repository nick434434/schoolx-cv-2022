import asyncio
from typing import Union, List
from base64 import b64decode

from json import loads as json_loads
import cv2
import numpy as np
from asyncio_mqtt import Client

from Materials.config import CONFIG
from Materials.Lecture_4.index import FeatureStorage
from Materials.Lecture_4.face_worker import FaceWorker


people = FeatureStorage(filename_index=CONFIG["faiss_index_path"], filename_users=CONFIG["faiss_users_path"],
                        num_features=CONFIG["space_size"])
deciding_threshold = CONFIG["threshold"]


def get_faces_and_features(image: Union[str, np.ndarray]):
    if isinstance(image, str):
        nparr = np.frombuffer(b64decode(image), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    face_processor = FaceWorker(image)
    face_processor.run()

    return face_processor.faces, face_processor.features


def process_message(data) -> List[str]:
    """
    Process incoming frame: find faces, check if there are 'our' people and return list of their usernames
    :param data: message from mqtt (including payload, etc.)
    :return: List of surely found people using their usernames
    """
    faces, features = get_faces_and_features(data)
    users_found = []

    if len(features) > 0 and features[0].shape == (128,):
        features = np.array(features)
        search_results = people.search(features, 1)

        for distance, faiss_id, username in search_results:
            if distance < deciding_threshold:
                users_found.append(username)

        return users_found


async def run():
    async with Client(CONFIG["mqtt_hostname"]) as client:
        async with client.filtered_messages("cameras/+") as messages:
            await client.subscribe("cameras/+")
            async for message in messages:
                print(message.topic, message.payload.decode())
                print(process_message(message.payload.decode()))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
