from dataclasses import dataclass
from typing import Optional
from dataclass_wizard import YAMLWizard

@dataclass
class FloodFill:
    sure_foreground_radius:float
    probable_foreground_radius:float
    probable_background_radius:float
    n_iter:int
    
@dataclass
class ImageEnhancement:
    felzenszwalb:bool
    refined:bool
    median_kernel_size:int
    gaussian_kernel_size:int
    kmeans_iters:int 
    glare_threshold:int

@dataclass
class Daugman:
    min_radius_ratio:float
    max_radius_ratio:float
    min_radius_pixels:int
    radius_step:int
    points_step:int
    reduce_points:bool
    force_square:bool

@dataclass 
class Segmentation:
    segmentor:str
    model_path:str
    image_enhancement:ImageEnhancement
    daugman:Daugman
    flood_fill:FloodFill

@dataclass
class Detection:
    detector:str
    minimum_height:int

@dataclass
class Config(YAMLWizard):
    debug:bool
    run_name:Optional[str]
    alpha:float
    output_dir:str
    data_dir:str
    ground_truth_dir:str
    detection:Detection
    segmentation:Segmentation
  

if __name__ == "__main__":
    conf_file = './config.yaml'
    conf = Config.from_yaml_file(conf_file)
    a = 3