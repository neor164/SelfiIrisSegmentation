import cv2
from numpy import ndarray
import os.path as osp
from .det_utils import filter_by_face
from config import Detection
class HaarCascadeEye:
    def __init__(self, config:Detection) -> None:
        self.config = config
        detector_dir = osp.dirname(__file__)
        self.eye_cascade = cv2.CascadeClassifier(osp.join(detector_dir,'haarcascade_eye.xml'))
        self.face_cascade = cv2.CascadeClassifier(osp.join(detector_dir,'haarcascade_frontalface_default.xml'))
    def detect(self, image: ndarray) -> ndarray:

        
        eyes = self.eye_cascade.detectMultiScale(image, scaleFactor = 1.2,
                                    minNeighbors = 5)
        faces = self.face_cascade.detectMultiScale(image, scaleFactor = 1.2,
                                    minNeighbors = 5)

        ans = filter_by_face(eyes,faces,self.config)
        
        return ans


