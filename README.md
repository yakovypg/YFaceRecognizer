<h1 align="center">YFaceRecognizer</h1>
<p align="center">
  <img alt="yfacerecognizer" height="200" src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExaGx5MzgxbHdjNXRkZ3Ywb3VmdjI0emN5ODMza3VoNHJsenM5cXFpMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/TB10Kv09bURxgLnsPX/giphy.gif" />
</p>

<p align="center">
  <a href="https://github.com/yakovypg/YFaceRecognizer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-darkyellow.svg" alt="license" />
  </a>
  <img src="https://img.shields.io/badge/Version-1.0.0-red.svg" alt="version" />
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="python" />
</p>

## About
**YFaceRecognizer** is a tool that will help you easily build a face recognition system. You can use basic scripts to build your own system or use a ready-made CLI application.

## Table of contents
*    [Quick Start](#quick-start)
     * [Download Models](#download-models)
     * [Add Known Faces](#add-known-faces)
     * [Run Tool](#run-tool)
     * [Create Face Encoding](#create-face-encoding)
*    [Perform Tests](#perform-tests)
*    [License](#license)

## Quick Start
Follow these steps:
- [Download](#download-model) `shape_predictor_68_face_landmarks.dat`.
- [Create](#add-known-faces) folder for storing known faces and move the relevant images or json files into it.
- Clone repository.
    ```
    git clone https://github.com/yakovypg/YFaceRecognizer.git
    ```
- Open the following folder.
    ```
    cd YFaceRecognizer
    ```
- Create virtual environment.
    ```
    python3 -m venv .venv
    ```
- Activate virtual environment.
    ```
    source .venv/bin/activate
    ```
- Install required packages.
    ```
    pip install -r requirements.txt
    ```
- Open `src` folder and [run](#run-tool) `main.py`.

### Download Models
The following models and weights need to be downloaded:
1. `shape_predictor_68_face_landmarks.dat` (can be downloaded [here](https://github.com/davisking/dlib-models)).
2. `antispoofing_model.json` (can be downloaded [here](https://github.com/prabhat-123/Face_Antispoofing_System/tree/main/antispoofing_models)).
3. `antispoofing_model.h5` (can be downloaded [here](https://github.com/prabhat-123/Face_Antispoofing_System/tree/main/antispoofing_models)).

You can download models and weights from the provided links or anywhere else. To do this in one command, run the `download_models.sh` [script](download_models.sh).

```
./download_models.sh
```

### Add Known Faces
To add a new face, you need to add an image or [face encoding](#create-face-encoding) to the folder with known faces. Make sure the image contains only one face. If the folder doesn't exist yet, you can create it in any location you like. Next, specify this folder when running the tool.

You can download sample face dataset using the `download_faces.sh` [script](download_faces.sh).

```
./download_faces.sh
```

### Run Tool
> **Note**
> Don't close images or video stream with the cross. Instead, press the `q` key on your keyboard.

You should specify path to the `shape_predictor_68_face_landmarks.dat` and path to the folder containing known faces. It is not necessary to specify the paths to the input files and the output path.

If you don't specify the paths to the input files, the tool will read video stream from your webcam.

```
python3 main.py -m /path/to/shape_predictor_68_face_landmarks.dat -k /path/to/folder_with_known_faces
```

If only one input file is specified, the output path can be the path the file or directory where output image will be saved.

```
python3 main.py -m /path/to/shape_predictor_68_face_landmarks.dat -k /path/to/folder_with_known_faces -i /path/to/input_image -o /path/to/output_image_or_directory
```

If multiple input files are specified, the output path can only be the path to the directory where output image will be saved.

```
python3 main.py -m /path/to/shape_predictor_68_face_landmarks.dat -k /path/to/folder_with_known_faces -i /path/to/input_image_1 /path/to/input_image_2 -o /path/to/output_directory
```

You can omit the output path. In this case, the output images will not be saved, but will be displayed on the screen instead.

```
python3 main.py -m /path/to/shape_predictor_68_face_landmarks.dat -k /path/to/folder_with_known_faces -i /path/to/input_image_1 /path/to/input_image_2
```

You can enable anti-spoofing mode. To do this, specify the `--spoof-check` flag. You also should specify the path to the model `antispoofing_model.json` and path to its weights `antispoofing_model.h5`.

```
python3 main.py -m /path/to/shape_predictor_68_face_landmarks.dat -k /path/to/folder_with_known_faces --spoof-check --spoof-model /path/to/antispoofing_model.json --spoof-model-weights /path/to/antispoofing_model.h5
```

### Create Face Encoding
You can store known faces not only as an image, but also as a json object containing a name and an encoding (embedding). You can create this file using `create_encoding.py`.

```
python3 create_encoding.py -i /path/to/input_image -o /path/to/output_json -n "Person Name"
```

## Perform Tests
You can test `FaceRecognizer` class using `test.py` script.

```
python3 test.py -m /path/to/shape_predictor_68_face_landmarks.dat -k /path/to/folder_with_known_faces -o /path/to/folder_where_results_will_be_saved -i /path/to/folder_with_input_images -n "Expected Name"
```

For example, if you have the following project structure:
```
YFaceRecognizer/
    KnownFaces/
        james.png
        mark.jpg
        robert.json
        ...
    src/
        test.py
        ...
    TestFaces/
        Mark/
            mark-1.jpg
            mark_photo.jpg
            mark99.jpg
    shape_predictor_68_face_landmarks.dat
```

The command will be like:
```
python3 test.py -m ../shape_predictor_68_face_landmarks.dat -k ../KnownFaces -o ../Results -i ../TestFaces/Mark -n "mark"
```

And after executing this command a project structure will be as follows:
```
YFaceRecognizer/
    KnownFaces/
        james.png
        mark.jpg
        robert.json
        ...
    Results/
        Mark/
            mark-1_recognized.jpg
            mark-2_recognized.jpg
            mark-3_recognized.jpg
    src/
        test.py
        ...
    TestFaces/
        Mark/
            mark-1.jpg
            mark_photo.jpg
            mark99.jpg
        ...
    shape_predictor_68_face_landmarks.dat
```

## License
The project is available under the [MIT](LICENSE) license.
