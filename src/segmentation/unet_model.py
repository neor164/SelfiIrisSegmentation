""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import numpy as np
from numpy import ndarray
from torchvision.transforms import ToTensor
import cv2
from  .seg_utils import IrisData

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





class IrisUnet:
    def __init__(self, checkpoint_path:str) -> None:
        self.model = UNet(1,1)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        self.img_size = (160, 60)
        self.to_tensor = ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, im:ndarray) ->np.ndarray:

        image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        image = self.to_tensor(cv2.resize(image,self.img_size)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            pred_mask = self.model(image)
        return pred_mask.cpu().numpy()

    def segment(self, image:ndarray, eye_bb:list[int]) ->IrisData:
        x, y, w, h = eye_bb
        eye = image[y : y + h, x : x + w]
        mask = (self.__getitem__(eye).squeeze()>0).astype(np.uint8)* 255
        mask = cv2.resize(mask, (w, h))
        return IrisData(None,eye_bb,eye,mask)
