import cv2 as cv
import numpy as np

from tensorflow.keras.models import model_from_json


class LivenessDetector(object):
    def __init__(self, model_path, model_weights_path):
        self.model_path = model_path
        self.model_weights_path = model_weights_path

    def is_face_spoof(self, face_image, min_probability=0.6):
        model_json = None

        with open(self.model_path, "r") as file:
            model_json = file.read()

        model = model_from_json(model_json)
        model.load_weights(self.model_weights_path)

        resized_face_image = cv.resize(face_image, (160, 160))
        resized_face_image = resized_face_image.astype("float") / 255.0
        resized_face_image = np.expand_dims(resized_face_image, axis=0)

        prediction = model.predict(resized_face_image)[0]

        return prediction >= min_probability
