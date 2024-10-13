import argparse

from face_info import FaceInfo

def _configure_parser():
    parser = argparse.ArgumentParser(
        prog="CreateEncoding",
        description="script for creating json file that can be used as known face for FaceRecognizer")

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="path to the input image")
    
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="path to the output file")

    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="name of the person")

    return parser

if __name__ == "__main__":
    parser = _configure_parser()
    args = parser.parse_args()

    FaceInfo.create_face_encoding(args.input, args.output, args.name)
