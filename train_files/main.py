from unet_model import UNet
import numpy as np      
import torch
unet = UNet(1,1)
import matplotlib.pyplot as plt     
import cv2  
from dataset import SythIris
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loss import dice_loss
from evaluate import evaluate
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import os
from datetime import datetime

batch_size = 32
model = UNet(1,1)
check_point_path = 'checkpoints/23-12-2022/checkpoint_epoch8.pth'
now = datetime.now()
run_name_dir = now.strftime("%m-%d-%Y_%H-%M")
dir_checkpoint = Path(os.path.join('./checkpoints/', run_name_dir))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
model = model.to(memory_format=torch.channels_last)
criterion = nn.BCEWithLogitsLoss()
data_seg = '/mnt/external/storage/dataset/synth-eye/segmentation/49985.png'
root_dir = '/mnt/external/storage/dataset/syth_eye_wallmart/'
dataset = SythIris(root_dir)
val_percent = 0.05
train_percent = 0.5

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
n_train = int(len(dataset) * train_percent)
n_p = int(len(dataset) - n_val - n_train)
train_set, val_set, p_set = random_split(dataset, [n_train, n_val, n_p], generator=torch.Generator().manual_seed(0))
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
epochs: int = 15
learning_rate: float = 1e-5
save_checkpoint: bool = True
img_scale: float = 0.5
amp: bool = False
weight_decay: float = 1e-8
momentum: float = 0.999
gradient_clipping: float = 1.0

optimizer = optim.RMSprop(model.parameters(),
                            lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
global_step = 0

# 5. Begin training
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for images, true_masks  in train_loader:
            # images, true_masks = batch['image'], batch['mask']

            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks.float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.squeeze(1).float(), multiclass=False)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
   
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)

            

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
