import os
import math

import cv2 as cv
import dlib as dl
import face_recognition as fr

from pathlib import Path
from face_info import FaceInfo


class FaceRecognizer(object):
    def __init__(self, shape_predictor_path):
        self.face_detector = dl.get_frontal_face_detector()
        self.shape_predictor = dl.shape_predictor(shape_predictor_path)

        self.known_face_encodings = []
        self.known_face_names = []

        self.landmarks_count = 68

        self.face_landmark_color = (255, 0, 0)
        self.face_rectangle_color = (0, 255, 0)
        self.name_color = (255, 255, 255)

    def add_known_faces(self, directory_path):
        if not os.path.isdir(directory_path):
            raise RuntimeError(f"error: '{directory_path}' is not directory")

        file_names = os.listdir(directory_path)

        for file_name in file_names:
            path = os.path.join(directory_path, file_name)
            file_extension = Path(path).suffix

            if file_extension == ".json":
                self.add_known_face_from_json(path)
            elif file_extension in (".png", ".jpg", ".jpeg"):
                self.add_known_face_from_image(path)
            else:
                raise RuntimeError(f"error: file with extension '{file_extension}' is not supported")

    def add_known_face_from_image(self, image_path):
        if not os.path.isfile(image_path):
            raise RuntimeError(f"error: '{image_path}' is not file")

        image = fr.load_image_file(image_path)
        face_locations = fr.face_locations(image)

        if not face_locations:
            raise RuntimeError(f"error: there is no face in the image '{image_path}'")

        encodings = fr.face_encodings(image, face_locations)

        if len(encodings) != 1:
            raise RuntimeError(f"error: image '{image_path}' has several faces")

        name = Path(image_path).stem
        encoding = encodings[0]

        self.known_face_names.append(name)
        self.known_face_encodings.append(encoding)

    def add_known_face_from_json(self, json_path):
        if not os.path.isfile(json_path):
            raise RuntimeError(f"error: '{json_path}' is not file")

        face = FaceInfo.get_from_json(json_path)

        self.known_face_names.append(face.name)
        self.known_face_encodings.append(face.encoding)

    def add_face_info_to_frame(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        face_locations = fr.face_locations(frame_rgb)
        face_encoding = fr.face_encodings(frame_rgb, face_locations)

        names = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
            name = self.get_face_name(face_encoding)
            names.append(name)

            self.add_face_name_to_frame(frame, name, left, top)
            self.add_face_rectangle_to_frame(frame, left, top, right, bottom)
            self.add_face_landmarks_to_frame(frame, frame_gray, left, top, right, bottom)

        return (frame, names)

    def add_face_name_to_frame(self, frame, name, face_left, face_top):
        FONT_SCALE = 2e-3
        THICKNESS_SCALE = 1e-3
        TEXT_Y_OFFSET_SCALE = 1e-2
        TEXT_MARGIN_RIGHT = 5

        frame_height, frame_width, _ = frame.shape

        text = name
        text_font = cv.FONT_HERSHEY_TRIPLEX
        text_font_scale = min(frame_width, frame_height) * FONT_SCALE
        text_color = self.name_color
        text_thickness = math.ceil(min(frame_width, frame_height) * THICKNESS_SCALE)

        text_size = cv.getTextSize(text, fontFace=text_font, fontScale=text_font_scale, thickness=text_thickness)
        text_width = text_size[0][0]
        default_free_space = frame_width - face_left - TEXT_MARGIN_RIGHT

        text_bottom_left_x = (
            face_left
            if default_free_space > text_width
            else (face_left - (text_width - default_free_space))
        )

        text_bottom_left_y = face_top - int(frame_height * TEXT_Y_OFFSET_SCALE)
        text_bottom_left = (text_bottom_left_x, text_bottom_left_y)

        cv.putText(frame, text, text_bottom_left, text_font, text_font_scale, text_color, text_thickness)

    def add_face_rectangle_to_frame(self, frame, face_left, face_top, face_right, face_bottom):
        THICKNESS_SCALE = 1e-3

        frame_height, frame_width, _ = frame.shape

        rectangle_start_point = (face_left, face_top)
        rectangle_end_point = (face_right, face_bottom)
        rectangle_color = self.face_rectangle_color
        rectangle_thickness = math.ceil(min(frame_width, frame_height) * THICKNESS_SCALE)

        cv.rectangle(frame, rectangle_start_point, rectangle_end_point, rectangle_color, rectangle_thickness)

    def add_face_landmarks_to_frame(self, frame, frame_gray, face_left, face_top, face_right, face_bottom):
        face = dl.rectangle(face_left, face_top, face_right, face_bottom)
        landmarks = self.shape_predictor(frame_gray, face)

        for i in range(self.landmarks_count):
            center_x = landmarks.part(i).x
            center_y = landmarks.part(i).y
            center = (center_x, center_y)

            self.add_face_landmark_to_frame(frame, center)

    def add_face_landmark_to_frame(self, frame, center):
        RADIUS_SCALE = 1e-3

        frame_height, frame_width, _ = frame.shape

        radius = math.ceil(min(frame_width, frame_height) * RADIUS_SCALE)
        color = self.face_landmark_color
        thickness = -1

        cv.circle(frame, center, radius, color, thickness)

    def get_face_name(self, face_encoding):
        matches = fr.compare_faces(self.known_face_encodings, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            return self.known_face_names[first_match_index]

        return "Unknown"
