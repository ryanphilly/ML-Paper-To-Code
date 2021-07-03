
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.nn.modules.activation import ReLU
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
  def __init__(self, noise_dim, output_channels):
    super(Gen, self).__init__()
    self.linear = nn.Sequential(
      nn.Linear(noise_dim, 7*7*256),
      nn.BatchNorm1d(7*7*256),
      nn.ReLU(),
    )

    # strided covolution layers
    self.upsample = nn.Sequential(
      nn.ConvTranspose2d(256, 128, (5, 5)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, (5, 5), stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, output_channels, (5, 5), stride=2),
      nn.Tanh()
    )

  def forward(self, data):
    data = self.linear(data).view(-1, 256, 7, 7)
    return self.upsample(data)

def noise(batch_size, n_features=100, device='cuda'):
  """creates a noise matrix for a given batch size"""
  return Variable(torch.randn(batch_size, n_features)).to(device)

noisee = noise(64)
generator = Gen(noisee.shape[1], 1).to('cuda')
print(generator(noisee))