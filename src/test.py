import os
import argparse

import cv2 as cv

from pathlib import Path
from face_recognizer import FaceRecognizer

def _test(
        expected_person_name,
        known_faces_path,
        person_images_directory_path,
        results_path,
        shape_predictor_path):
    face_recognizer = FaceRecognizer(shape_predictor_path)
    face_recognizer.add_known_faces(known_faces_path)

    person_image_names = os.listdir(person_images_directory_path)
    _, person_images_directory_name = os.path.split(person_images_directory_path)

    processed_images_count = 0
    images_count = len(person_image_names)

    errors = []

    for person_image_name in person_image_names:
        person_image_path = os.path.join(person_images_directory_path, person_image_name)

        image = cv.imread(person_image_path)
        image, actual_names = face_recognizer.add_face_info_to_frame(image)

        result_directory_path = os.path.join(results_path, person_images_directory_name)
        result_image_path = os.path.join(result_directory_path, person_image_name)

        Path(result_directory_path).mkdir(parents=True, exist_ok=True)
        cv.imwrite(result_image_path, image)

        if expected_person_name not in actual_names:
            errors.append(person_image_name)
        
        processed_images_count += 1
        print(f"images processed: {processed_images_count} out of {images_count}")

    errors_count = len(errors)
    errors_percentage = "%.2f" % (errors_count / images_count * 100)
    
    print()
    print(f"#images: {images_count}")
    print(f"#errors: {errors_count} ({errors_percentage}%)")

    if len(errors) > 0:
        print()
        print("errors:")

        for i, name in zip(range(1, len(errors) + 1), errors):
            print(f"{i}. {name}")

def _configure_parser():
    parser = argparse.ArgumentParser(
        prog="FaceRecognizerTest",
        description="script for testing FaceRecognizer")
    
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="path to the shape_predictor_68_face_landmarks.dat")
    
    parser.add_argument(
        "-k",
        "--known-faces",
        required=True,
        help="path to the directory with faces that must be registered in the system")

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="path to the directory where the results will be saved")
    
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="path to the directory containing faces of one person that must be recognized")

    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="expected name of the preson")

    return parser

if __name__ == "__main__":
    parser = _configure_parser()
    args = parser.parse_args()
    
    _test(args.name, args.known_faces, args.input, args.output, args.model)
