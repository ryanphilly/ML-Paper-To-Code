import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.optim import optimizer

LEARNING_RATE = 2e-4
MOMMENTUM = 0.5


def weights_init(module):
  classname = module.__class__.name__
  if classname.find('Conv') != -1:
    nn.init.normal_(module.weight.data, 0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(module.weight.data, 0.0, 0.02)
    nn.init.constant_(module.bias.data, 0)


class Gen(nn.Module):
  def __init__(self, noise_dim, layer_dims):
    self.seq = nn.Sequential(

    )



opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)