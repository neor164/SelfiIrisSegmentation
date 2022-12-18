from typing import Tuple
from numpy import ndarray
import cv2
import numpy as np
from .seg_utils import alpha_blend_image
from config import FloodFill


def flood_fill(image:np.ndarray, eye_coor:Tuple[int,int],eye_radius:int, config:FloodFill) -> ndarray:
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    sure_fg_radius = int(eye_radius * config.sure_foreground_radius)
    probable_fg_radius = int(eye_radius * config.probable_foreground_radius)
    probable_bg_radius = int(eye_radius * config.probable_background_radius)
    probable_bg = 2
    probable_fg = 3
    sure_fg = 1
    cv2.circle(mask, eye_coor, probable_bg_radius, probable_bg, -1)
    cv2.circle(mask, eye_coor, probable_fg_radius, probable_fg, -1)
    cv2.circle(mask, eye_coor, sure_fg_radius, sure_fg, -1)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,config.n_iter,cv2.GC_INIT_WITH_MASK)
    mask[(mask==2) ] = 0
    mask[(mask!=0) ] = 255
    return mask 







if __name__ == '__main__':
    image = cv2.imread('gt/source/Selfie_Alejandra_age_32_1.jpg')
    crop_bb = [496,470, 825, 801]
    cropped_image = image[crop_bb[1]:crop_bb[3],crop_bb[0]:crop_bb[2]]
    eye_coor = [666, 652]
    eye_radius = 40
    
    eye_coor = eye_coor[0] - crop_bb[0],eye_coor[1] - crop_bb[1]  
    mask = flood_fill(cropped_image, eye_coor, eye_radius)
    new_cropped_image = alpha_blend_image(cropped_image, mask)
    image[crop_bb[1]:crop_bb[3],crop_bb[0]:crop_bb[2]] = new_cropped_image 
    plt.imshow(image)
    plt.show()