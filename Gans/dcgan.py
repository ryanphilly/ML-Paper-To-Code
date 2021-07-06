
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from .util import train, noise, tanh_transform

LEARNING_RATE = 2e-4

def weights_init(module):
  classname = module.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(module.weight.data, 0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(module.weight.data, 1.0, 0.02)
    nn.init.constant_(module.bias.data, 0)


class Gen(nn.Module):
  def __init__(self, noise_dim, output_channels=1):
    super(Gen, self).__init__()
    self.linear = nn.Sequential(
      nn.Linear(noise_dim, 7*7*256),
      nn.BatchNorm1d(7*7*256),
      nn.ReLU(),
    )

    # strided covolution layers
    self.upsample = nn.Sequential(
      nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),
      nn.Tanh()
    )

  def forward(self, data):
    data = self.linear(data).view(-1, 256, 7, 7)
    return self.upsample(data)


class Dis(nn.Module):
  def __init__(self, in_channels=1):
    super(Dis, self).__init__()
    self.conv_seq = nn.Sequential(
      nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
      nn.LeakyReLU(0.2),
  
      nn.Conv2d(64, 128, 4, stride=2, padding=1),
      nn.LeakyReLU(0.2),
      nn.BatchNorm2d(128),


      nn.Conv2d(128, 256, 5, stride=1, padding=2),
      nn.LeakyReLU(0.2),
      nn.BatchNorm2d(256),
    )

    self.lin = nn.Sequential(
      nn.Linear(256*7*7, 1),
      nn.Sigmoid()
    )

  def forward(self, image):
    return self.lin(self.conv_seq(image)
      .view((-1, 256*7*7)))



noisee = noise(64)

data = MNIST('./data/', train=True, transform=tanh_transform)
trainloader = DataLoader(data, batch_size=100, shuffle=True)

generator = Gen(noisee.shape[1], 1).to('cuda')
discriminator = Dis().to('cuda')
generator.apply(weights_init)
discriminator.apply(weights_init)

opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

loss = nn.BCELoss()

train(trainloader, generator, discriminator, opt_g, opt_d, loss, 250, 1)