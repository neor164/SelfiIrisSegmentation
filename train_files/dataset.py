import pandas as pd
from glob import glob
import os.path as osp
from torchvision import datasets 
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torchvision.transforms import ToTensor



root_dir = '/mnt/external/storage/dataset/synth-eye'
seg_sub_dir = 'segmentation'
im_sub_sir = 'lit'
l = glob(osp.join(root_dir,seg_sub_dir,'*.png'))


class SythIris(Dataset):
    def __init__(self, root_dir:str, transform=None, target_transform=None):
        self.img_labels = osp.join(root_dir,'masks')
        self.img_dir = osp.join(root_dir,'imgs')
        self.transform = transform
        self.target_transform = target_transform
        self.images_paths = glob(osp.join(self.img_dir, '*.png'))
        self.eye_lid_values = [204,158,0]
        self.eye_case = [78, 204, 156]
        self.bias = 8
        self.img_size = (160, 60)
        self.to_tensor = ToTensor()
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        image = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
        image = self.to_tensor(cv2.resize(image,self.img_size))
        fname = osp.basename(img_path)
        label_path = osp.join(self.img_labels, fname)
        mask = cv2.cvtColor(cv2.imread(label_path),cv2.COLOR_BGR2GRAY)
        mask  =  self.to_tensor(cv2.resize(mask, self.img_size))
       
        return image, mask


if __name__  == "__main__":
    root_dir = '/mnt/external/storage/dataset/syth_eye_wallmart/'
    dataset = SythIris(root_dir)
    for i in tqdm(range(45510,len(dataset))):
        im, mask  = dataset[i]    
