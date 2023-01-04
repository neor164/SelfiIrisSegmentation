from model import LightIrisSegmentation
import numpy as np      
import torch
import matplotlib.pyplot as plt     
import cv2  
from dataset import SythIris
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loss import DiceLoss
from evaluate import evaluate
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import os
from datetime import datetime



batch_size = 32
model =  LightIrisSegmentation(1,1)
# check_point_path = 'checkpoints/23-12-2022/checkpoint_epoch8.pth'
now = datetime.now()
run_name_dir = now.strftime("%m-%d-%Y_%H-%M")
dir_checkpoint = Path(os.path.join('./checkpoints/', run_name_dir))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCEWithLogitsLoss()
alpha = 0.1

data_seg = '/mnt/external/storage/dataset/synth-eye/segmentation/49985.png'
root_dir = '/mnt/external/storage/dataset/syth_eye_wallmart/'
dataset = SythIris(root_dir)
val_percent = 0.05
train_percent = 0.5
dice_loss = DiceLoss()
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
epochs: int = 15
learning_rate: float = 1e-3
save_checkpoint: bool = True
optimizer = optim.Adam(model.parameters())







for epoch in range(1, epochs + 1):

    model.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for images, true_masks  in train_loader:
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)

            masks_pred = model(images) 
            l1 = criterion(masks_pred, true_masks.float())
            l2 = dice_loss(masks_pred, true_masks)
            loss = alpha * l1 + (1-alpha) * l2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.update(images.shape[0])
            pbar.set_postfix(**{'loss (batch)': loss.item()})
    pbar.set_postfix(**{'loss (batch)': epoch_loss/len(train_loader)})

    model.eval()
    num_val_batches = len(val_loader)
    dice_score = 0

    # iterate over the validation set
    with torch.no_grad():
        for image, mask_true in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = model(image)

            # compute the Dice score
            dice_score += dice_loss(mask_pred, mask_true)
            # else:
            #     assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            #     # convert to one-hot format
            #     mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    model.train()
    dice_score / max(num_val_batches, 1)
    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        # state_dict['mask_values'] = dataset.mask_values
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))