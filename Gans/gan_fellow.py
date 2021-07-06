# Goals
# Create a Generator network that learns the underliying data distribution
# and generates a sample.

# Create a Discriminator network that estimates the probability that the 
# sample came from the training data rather than the Generator

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from .util import train, tanh_transform

NUM_NOISE_FEATURES = 128
GEN_OUTPUT_FEATURES = 28 * 28

DIS_OUTPUT_FEATURES = 1
DIS_INPUT_FEATURES = GEN_OUTPUT_FEATURES

DEVICE = 'cuda:0'

class Generator(nn.Module):
  def __init__(self, num_noise_features=NUM_NOISE_FEATURES, num_out_features=GEN_OUTPUT_FEATURES):
    super(Generator, self).__init__()
    self.seq = nn.Sequential(
      nn.Linear(num_noise_features, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, num_out_features),
      nn.Tanh()
    )

  def forward(self, noise):
    return self.seq(noise).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
  def __init__(self, in_channels=DIS_INPUT_FEATURES, out_channels=DIS_OUTPUT_FEATURES):
    super(Discriminator, self).__init__()
    self.seq = nn.Sequential(
      nn.Linear(in_channels, 1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3),
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3),
      nn.Linear(256, out_channels),
      nn.Sigmoid()
    )
  
  def forward(self, img_vec):
    img_vec = img_vec.view(-1, 28*28)
    return self.seq(img_vec)


data = MNIST('./data/', train=True, transform=tanh_transform)
trainloader = DataLoader(data, batch_size=100, shuffle=True)

G = Generator(NUM_NOISE_FEATURES, GEN_OUTPUT_FEATURES).to(DEVICE)
G_OPT = torch.optim.Adam(G.parameters(), lr=2e-4)
D = Discriminator(DIS_INPUT_FEATURES, DIS_OUTPUT_FEATURES).to(DEVICE)
D_OPT = torch.optim.Adam(D.parameters(), lr=2e-4)

LOSS = nn.BCELoss()

train(trainloader, G, D, G_OPT, D_OPT, LOSS, 250, 1)




