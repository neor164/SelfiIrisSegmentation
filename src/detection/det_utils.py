from copy import deepcopy
from typing import Optional, Tuple
import numpy as np
from config import Detection



def filter_by_face(eyes:np.ndarray, faces:np.ndarray, detection_config:Detection) -> Optional[Tuple[np.ndarray,  np.ndarray]]:

    if len(faces) >= 1:
    
        faces = sorted(faces, key=lambda face: face[2] * face[3],reverse=True)
        face = faces[0]
        left_quad = deepcopy(face)
        left_quad[2:] = left_quad[2:]//2
        right_quad = deepcopy(face)
        right_quad[0] += right_quad[2]//2
        right_quad[2:] = right_quad[2:]//2
        left_eyes = [eye for eye in eyes if inside_face(eye, left_quad, detection_config.minimum_height) ]
        right_eyes =  [eye for eye in eyes if inside_face(eye, right_quad, detection_config.minimum_height) ]
        if len(left_eyes) >=1 and len(right_eyes) >=1:
            left_eye = sorted(left_eyes, key=lambda eye: eye[2] * eye[3],reverse=True)[0]
            right_eye = sorted(right_eyes, key=lambda eye: eye[2] * eye[3],reverse=True)[0]

            return left_eye, right_eye


def inside_face(eye_bb:np.ndarray, face_quad:np.ndarray, min_height:int=40) -> bool:
    x_check = eye_bb[0] > face_quad[0] and (eye_bb[0] + eye_bb[2]) < face_quad[0] + face_quad[2]
    y_check = (eye_bb[1] + eye_bb[3])  < (face_quad[1] +face_quad[3]) * 1.2   and (eye_bb[1]) > (face_quad[1]) * 1.2
    size_check = eye_bb[3] > min_height
    return x_check and y_check and size_check