import cv2 as cv

from face_recognizer import FaceRecognizer

class VideoFaceRecognizer(FaceRecognizer):
    def __init__(self, shape_predictor_path, video_stream):       
        super().__init__(shape_predictor_path)
        self.video_stream = video_stream
    
    def capture_video_frame(self):       
        return self.video_stream.read()

    def release_resources(self):
        self.video_stream.release()
        cv.destroyAllWindows()
