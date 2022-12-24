from typing import Optional, Tuple
import numpy as np

import cv2
import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.segmentation import felzenszwalb
from config import ImageEnhancement
from dataclasses import dataclass


@dataclass
class DaugmanData:
    radius:int
    # x in face image coordinates
    x_center:int
    # y in face image coordinates
    y_center:int
    # processed patch used to improve daugman performance
    enhanced_patch:np.ndarray


@dataclass
class IrisData:
    daugman_data:Optional[DaugmanData]
    patch_bounding_box:Tuple[int,int,int,int]
    patch_im:np.ndarray
    mask:np.ndarray



def get_square_bounding_box(bb:np.ndarray) -> np.ndarray:

    if bb[2]!=bb[3]:
        diff = bb[2]//2 - bb[3]//2 
        bb[1]  -= diff
        bb[3] = bb[2]
 
    return bb



def get_refined_points(img:np.ndarray, config:ImageEnhancement) ->np.ndarray:
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, config.kmeans_iters, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    # res = center[label.flatten()]
    label = (label.reshape((img.shape[0], img.shape[1])) == 1).astype(np.uint8) * 255
    # res = binary_fill_holes(label)

    all_points = np.argwhere(label)
    return all_points

def clean_gray_image(img:np.ndarray, config:ImageEnhancement) -> np.ndarray:
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[img >  config.glare_threshold] = 255
    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    dst = median_filter(dst, size=config.median_kernel_size)
    dst = gaussian_filter(dst, sigma=config.gaussian_kernel_size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(dst)
   
    if config.felzenszwalb: 
        segments = felzenszwalb(cl1, scale=100, sigma=0.2, min_size=cl1.size//8)
        # segments_slic = slic(gray_eye, n_segments=5, sigma=0,
        #      start_label=1)
        nsegments = np.unique(segments)
        for segment in nsegments:
            cl1[segments==segment] = cl1[segments==segment].mean()


    return cl1



def alpha_blend_image(image:np.ndarray,mask:np.ndarray, alpha:float=0.8, thresh=75)->np.ndarray:
    new_image = image.copy()
    new_image[:,:,2][mask>thresh] = image[:,:,2][mask>thresh] *(1 - alpha) + mask[mask>thresh] * (alpha) 
    return new_image