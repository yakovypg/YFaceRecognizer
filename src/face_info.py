import os
import json
import codecs

import cv2 as cv
import numpy as np
import face_recognition as fr


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


class FaceInfo(object):
    def __init__(self, name, encoding):
        self.name = name
        self.encoding = encoding

    def save_to_json(self, json_path):
        json.dump(
            self.__dict__,
            codecs.open(json_path, "w", encoding="utf-8"),
            cls=_NumpyEncoder)

    @staticmethod
    def create_face_encoding(image_path, json_path, person_name):
        if not os.path.isfile(image_path):
            raise RuntimeError(f"error: '{image_path}' is not file")

        image = cv.imread(image_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        face_locations = fr.face_locations(image_rgb)
        face_encoding = fr.face_encodings(image_rgb, face_locations)[0]

        face_info = FaceInfo(person_name, face_encoding)
        face_info.save_to_json(json_path)

    @staticmethod
    def get_from_json(json_path):
        if not os.path.isfile(json_path):
            raise RuntimeError(f"error: '{json_path}' is not file")

        with open(json_path, "r") as reader:
            json_data = reader.read()
            json_object = json.loads(json_data)

            name = json_object["name"]
            encoding = np.array(json_object["encoding"], dtype=float)

            return FaceInfo(name, encoding)
