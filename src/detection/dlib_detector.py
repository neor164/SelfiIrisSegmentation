import dlib
import os.path as osp
from numpy import ndarray
from config import Detection


class DlibDetector:
    def __init__(self,config:Detection):
        detector_dir = osp.dirname(__file__)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(osp.join(detector_dir,"shape_predictor_68_face_landmarks.dat"))
        self.config = config
        
    def detect(self, image:ndarray) -> ndarray:
        detect=self.detector(image,1)
        if len(detect):
            shape=self.predictor(image,detect[0])
            xmin=shape.part(36).x
            xmax=shape.part(39).x
            ymin=shape.part(37).y
            ymax=shape.part(40).y
            left_eye = [xmin, ymin, xmax - xmin, ymax - ymin]
            xmin=shape.part(42).x
            xmax=shape.part(45).x
            ymin=shape.part(43).y
            ymax=shape.part(46).y
            right_eye = [xmin, ymin, xmax - xmin, ymax - ymin]
            if right_eye[3] > self.config.minimum_height and left_eye[3] > self.config.minimum_height:

                return left_eye, right_eye

