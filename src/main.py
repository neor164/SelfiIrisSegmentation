import cv2
from pathlib import Path
from detection.haarcascade_eye import HaarCascadeEye
from detection.dlib_detector import DlibDetector
from segmentation.daugman import Daugman
from segmentation.unet_model import IrisUnet
from segmentation.light_iris import LightIris

from segmentation.seg_utils import  alpha_blend_image,clean_gray_image
from segmentation.fill_iris import  flood_fill
from evaluation import evaluate_results
import os.path as osp
import numpy as np
from glob import glob
from config import Config
from datetime import datetime
from tqdm import tqdm
import sys

def main(config:Config):
    now = datetime.now()
    detector = DlibDetector(config.detection) if config.detection.detector == "dlib" else HaarCascadeEye(config.detection)
    # segmentor = Daugman(config.segmentation)
    if config.segmentation.segmentor == "unet":
        segmentor = IrisUnet(config.segmentation.model_path)
    elif config.segmentation.segmentor == "light_iris":
        # segmentor = IrisUnet(config.segmentation.model_path)
        segmentor = LightIris(config.segmentation.model_path)
    else:
        segmentor = Daugman(config.segmentation)
    run_name_dir = now.strftime("%m-%d-%Y_%H-%M") if config.run_name is None else config.run_name
    pattern = osp.join(config.data_dir, '*.jpg')
    images_paths = glob(pattern)
    out_mask_dir = osp.join(config.output_dir,run_name_dir, 'mask')
    out_blend_path = osp.join(config.output_dir,run_name_dir, 'blend')
    if config.debug:
        debug_root_dir = osp.join(config.output_dir,run_name_dir, 'debug')
        evaluation_dir = 'evaluation'     
        debug_images_path = osp.join(debug_root_dir, 'images')
        evaluation_path = osp.join(debug_root_dir, evaluation_dir)
        Path(evaluation_path).mkdir(parents=True,exist_ok=True)
        Path(debug_images_path).mkdir(parents=True,exist_ok=True)
        config.to_yaml_file(osp.join(evaluation_path, 'config.yaml'))
    Path(out_mask_dir).mkdir(parents=True,exist_ok=True)
    Path(out_blend_path).mkdir(parents=True,exist_ok=True)
    for img_path in tqdm(images_paths):
        img_name = osp.basename(img_path)
        subject_name = img_name.split('_')[1]
        image = cv2.imread(img_path)
        blend_im = image.copy()
        if config.debug:
            debug_im = image.copy()
        mask = np.zeros(image.shape[:2],dtype=np.uint8)
        ans = detector.detect(image)
        if ans is not None:
            yc = 0
            xc = 0
            for i , eye_bb in enumerate(ans):

                iris_data = segmentor.segment(image, eye_bb)
                # iris_data = segmentor()
                if iris_data is None:
                    continue
                new_cropped_image = alpha_blend_image(iris_data.patch_im, iris_data.mask,alpha=config.alpha)
                x,y ,w ,h = iris_data.patch_bounding_box
                out_path = osp.join(config.output_dir,run_name_dir, f'eye{i}_{subject_name}.png')
                cv2.imwrite(out_path,iris_data.patch_im)
                out_path = osp.join(config.output_dir,run_name_dir, f'mask{i}_{subject_name}.png')
                cv2.imwrite(out_path,iris_data.mask )


                # cv2.imwrite(out_path,mask)
                blend_im[y:y+h,x:x+w] = new_cropped_image
                mask[y:y+h,x:x+w] = iris_data.mask 
                if config.debug:
                    # daugman_data = iris_data.daugman_data
                    # center_point = daugman_data.x_center , daugman_data.y_center                    
                    wc = iris_data.patch_im.shape[1]*2
                    hc = iris_data.patch_im.shape[0]*2
                    # cv2.circle(debug_im,center_point , daugman_data.radius, (0,255,0), 3)
                    cv2.rectangle(debug_im, (x,y), (x+w,y+h), (255,0,), 3)
                    # debug_im[y:y+h,x:x+w] = new_cropped_image
                    new_gray = cv2.resize(cv2.cvtColor(iris_data.patch_im, cv2.COLOR_BGR2GRAY), (wc,hc))
                    # debug_im[yc:yc+hc,xc:xc+wc] =  np.stack((new_gray,)*3, axis=-1)
                    xc = wc
                    
        out_path = osp.join(out_blend_path, img_name)
        cv2.imwrite(out_path,blend_im)
        out_path = osp.join(out_mask_dir, img_name)
        cv2.imwrite(out_path,mask)
        if config.debug:
            debug_path = osp.join(debug_images_path, img_name)
            cv2.imwrite(debug_path,debug_im)

    evaluate_results(out_mask_dir, config.ground_truth_dir, evaluation_path, "results")



if __name__ == "__main__":
    output_dir = 'data'
    config_path = 'config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config = Config.from_yaml_file(config_path)
    main(config)
