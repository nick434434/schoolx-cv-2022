from typing import Union
from base64 import b64decode

from json import loads as json_loads
import cv2
import numpy as np
import aiomqtt

from Materials.config import CONFIG
from Materials.Lecture_4.index import FeatureStorage
from Materials.Lecture_4.face_worker import FaceWorker


class MainWorker:
    client = aiomqtt.Client(client_id="consumer")
    user_manager = FeatureStorage(filename_index=CONFIG["faiss_index_path"], filename_users=CONFIG["faiss_users_path"],
                                  num_features=CONFIG["space_size"])
    face_processor = FaceWorker()  # Init models before calling first time

    @staticmethod
    def get_faces_and_features(image: Union[str, np.ndarray]):
        if isinstance(image, str):
            image = cv2.imdecode(b64decode(image), cv2.IMREAD_UNCHANGED)

        face_processor = FaceWorker(image)
        face_processor.run()

        return face_processor.faces, face_processor.features

    @staticmethod
    def on_message(event: aiomqtt.MQTTMessage):
        payload = json_loads(event.payload.decode('utf-8'))
        image = payload["frame"]
        faces, features = MainWorker.get_faces_and_features(image)
        # TODO: test and finish

    # TODO: add all things to config
    def __init__(self, host=CONFIG["mqtt_hostname"], port=CONFIG["mqtt_port"], topic=CONFIG["mqtt_topic"]):
        self.host = host
        self.port = port
        self.topic = topic

    def connect(self):
        self.client.connect(self.host, self.port, keepalive=30)

    def setup_on_message(self):
        self.client.subscribe(self.topic)
        self.client.on_message = MainWorker.on_message
