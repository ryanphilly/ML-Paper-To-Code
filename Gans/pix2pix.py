import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision import transforms


'''
Paper-to-Code
Image-to-Image Translation with Conditional Adversarial Networks
      https://arxiv.org/pdf/1611.07004.pdf

    


  

Both generator and discriminator layers:
 convolution-BatchNorm-ReLu
'''