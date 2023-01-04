""" Parts of the U-Net model """

from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import Conv2dNormActivation
import numpy as np
from numpy import ndarray
from torchvision.transforms import ToTensor
import cv2
from  .seg_utils import IrisData


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel:int=1,
        stride:int=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                    in_channels, 
                    out_channels,
                    kernel_size=kernel, 
                    padding=kernel//2,
                    bias=False,stride=stride),
            nn.BatchNorm2d(out_channels))

    def forward(self, x)-> Tensor:
        return self.conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        mid_channels = in_channels // 4
        self.conv1 = ConvBlock(in_channels, mid_channels, stride=1)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d( mid_channels,   mid_channels, kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(mid_channels, out_channels, stride=1)
        
    

    def forward(self, x:Tensor)-> Tensor:
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x


class Psa(nn.Module):
    def __init__(self,channels:int, kernels:Tuple[int,int,int,int]) -> None:
        super().__init__()
        self.spc = Spc(channels, kernels)
        self.se = Se(channels)
    
    def forward(self,x:Tensor) -> Tensor:
    
        return self.se(self.spc(x))


class Spc(nn.Module):
    def __init__(self,channels:int, kernels:Tuple[int,int,int,int]) -> None:
        super().__init__()
        channels = channels//4
        self.c1 = ConvBlock(channels, channels, kernel=kernels[0])
        self.c2 = ConvBlock(channels, channels, kernel=kernels[1])
        self.c3 = ConvBlock(channels, channels, kernel=kernels[2])
        self.c4 = ConvBlock(channels, channels, kernel=kernels[3])



    def forward(self, x:Tensor)->Tensor:
        features:Tuple[Tensor, Tensor, Tensor,Tensor] = torch.split(x, x.shape[1]//4, dim=1) 
        o1 = self.c1(features[0])
        o2 = self.c2(features[1])
        o3 = self.c3(features[2])
        o4 = self.c4(features[3])
        return torch.concat((o1,o2,o3,o4),dim=1)


class Se(nn.Module):

    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x:Tensor)->Tensor:
        bs, c, _, _ = x.shape
        y:Tensor = self.squeeze(x).view(bs, c)
        y:Tensor = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)



class BottleNeck(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int = 6, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__() 
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x:Tensor) ->Tensor:
        return self.downsample(x)


class DeepEncoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.b1 = BottleNeck(in_channels, out_channels ,stride=2)
        self.b2 = BottleNeck(out_channels, out_channels ,stride=1)

    def forward(self, x:Tensor) ->Tensor:
        x = self.b1(x)
        fx = self.b2(x)
        return fx + x
 

class ShallowEncoder(nn.Module):
    def __init__(self, in_channels:int,out_channels:int, kernels:Tuple[int,int,int,int]):
        super().__init__()

        self.conv1 = ConvBlock(in_channels,out_channels,kernel=3,stride=2)
        self.down = Down(in_channels, out_channels)
        self.psa = Psa(out_channels,kernels)
        self.conv2 = ConvBlock(out_channels,out_channels,kernel=3,stride=1)

    def forward(self, x:Tensor) ->Tensor:
        fx = self.conv2(self.conv1(x))
        x = self.down(x)
        y = fx + x 
        fy = self.conv2(self.psa(y))
        return y + fy
        

class InitBlock(nn.Module):
    def __init__(self, in_channels:int=1,out_channels:int=64) -> None:
        super().__init__()

        self.f = nn.Sequential(
            ConvBlock(in_channels,out_channels,kernel=7,stride=2),
            nn.MaxPool2d(2)
        )

    def forward(self, x:Tensor) -> Tensor: 
        return self.f(x)
    
class FinalBlock(nn.Module):
    def __init__(self, in_channels:int=64,out_channels:int=1) -> None:
        super().__init__()

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d( in_channels,   in_channels // 2, kernel_size=2, stride=2),
            ConvBlock(in_channels // 2, in_channels // 4, kernel=3),
            nn.ConvTranspose2d( in_channels // 4,   out_channels, kernel_size=2, stride=2),

        )

    def forward(self, x:Tensor) -> Tensor:
        return self.final_block(x)


class LightIrisSegmentation(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        channels:List[int] = [64, 128, 160, 256]
        kernels = (3, 5, 9, 9)
        self.init_block = InitBlock(in_channels, channels[0])

        self.e1 = ShallowEncoder(channels[0], channels[0],kernels)
        self.e2 = ShallowEncoder(channels[0],channels[1],kernels)
        self.e3 = DeepEncoder(channels[1], channels[2])
        self.e4 = DeepEncoder(channels[2], channels[3])

        self.d4 = Up(channels[3], channels[2])
        self.d3 = Up(channels[2], channels[1])
        self.d2 = Up(channels[1], channels[0])
        self.d1 = Up(channels[0], channels[0],bilinear=False)
        self.final_block = FinalBlock(channels[0], out_channels)


    def forward(self, x:Tensor) ->Tensor:

        x = self.init_block(x)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        y4 = self.d4(x4) + x3
        y3 = self.d3(y4) + x2
        y2 = self.d2(y3) + x1
        y = self.d1(y2)
        return self.final_block(y)



class LightIris:
    def __init__(self, checkpoint_path:str) -> None:
        self.model = LightIrisSegmentation(1,1)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        self.img_size = (255, 255)
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

if __name__ == '__main__':
    batch__size  = 1 
    sample = torch.zeros((1,1, 256, 256))
    net = LightIrisSegmentation(1,1)
    out = net(sample)
