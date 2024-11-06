import os
import argparse

import cv2 as cv

from face_recognizer import FaceRecognizer
from video_face_recognizer import VideoFaceRecognizer
from liveness_detector import LivenessDetector


def _process_webcam_video(known_faces_path, shape_predictor_path, liveness_detector=None):
    video_stream = cv.VideoCapture(0)

    video_face_recognizer = VideoFaceRecognizer(shape_predictor_path, video_stream, liveness_detector)
    video_face_recognizer.add_known_faces(known_faces_path)

    _process_video(video_face_recognizer)

    video_face_recognizer.release_resources()


def _process_images(images_paths, known_faces_path, shape_predictor_path, output_path=None, liveness_detector=None):
    PROCESSED_IMAGE_MARK = "_recognized"

    face_recognizer = FaceRecognizer(shape_predictor_path, liveness_detector)
    face_recognizer.add_known_faces(known_faces_path)

    for image_path in images_paths:
        processed_image_path = output_path

        if output_path is not None and os.path.isdir(output_path):
            image_name, image_extension = os.path.splitext(os.path.basename(image_path))
            processed_image_name = f"{image_name}{PROCESSED_IMAGE_MARK}{image_extension}"
            processed_image_path = os.path.join(output_path, processed_image_name)

        _process_image(face_recognizer, image_path, processed_image_path)


def _process_video(video_face_recognizer):
    while True:
        result, frame = video_face_recognizer.capture_video_frame()

        if not result:
            print("warning: cannot get frame from video stream")
            continue

        frame, _, _ = video_face_recognizer.add_face_info_to_frame(frame)
        cv.imshow("video", frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break


def _process_image(face_recognizer, image_path, output_path=None):
    if not os.path.isfile(image_path):
        raise RuntimeError(f"error: '{image_path}' is not file")

    image = cv.imread(image_path)
    image, _, _ = face_recognizer.add_face_info_to_frame(image)

    if output_path is None:
        cv.imshow("image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        cv.imwrite(output_path, image)


def _verify_output_path(args):
    if (args.input is not None
            and len(args.input) > 1
            and args.output is not None
            and not os.path.isdir(args.output)):
        print("error: output path is not directory")
        exit(1)


def _verify_spoof_args(args):
    if args.spoof_check is not None and (args.spoof_model is None or args.spoof_model_weights is None):
        print("error: --spoof-check requires --spoof-model and --spoof-model-weights")
        exit(1)


def _configure_parser():
    parser = argparse.ArgumentParser(
        prog="YFaceRecognizer",
        description="face recognition system")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="path to the shape_predictor_68_face_landmarks.dat")

    parser.add_argument(
        "-k",
        "--known-faces",
        type=str,
        required=True,
        help="path to the directory with faces that must be registered in the system")

    parser.add_argument(
        "--spoof-check",
        action="store_true",
        default=False,
        required=False,
        help="checking a face for spoof")

    parser.add_argument(
        "--spoof-model",
        type=str,
        required=False,
        help="path to the antispoofing_model.json")

    parser.add_argument(
        "--spoof-model-weights",
        type=str,
        required=False,
        help="path to the antispoofing_model.h5")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        nargs="+",
        help="paths to the images in which faces must be recognized")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="path to the output image or directory where output images must be saved")

    return parser


if __name__ == "__main__":
    parser = _configure_parser()
    args = parser.parse_args()

    _verify_output_path(args)
    _verify_spoof_args(args)

    liveness_detector = (
        LivenessDetector(args.spoof_model, args.spoof_model_weights)
        if args.spoof_check
        else None
    )

    if args.input is None:
        _process_webcam_video(args.known_faces, args.model, liveness_detector)
    else:
        _process_images(args.input, args.known_faces, args.model, args.output, liveness_detector)
