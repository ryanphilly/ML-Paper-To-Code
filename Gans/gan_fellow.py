# Goals
# Create a Generator network that learns the underliying data distribution
# and generates a sample.

# Create a Discriminator network that estimates the probability that the 
# sample came from the training data rather than the Generator

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision import transforms

to_image = transforms.ToPILImage()

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


def noise(batch_size, n_features=128, device='cuda'):
  """creates a noise matrix for a given batch size"""
  return Variable(torch.randn(batch_size, n_features)).to(device)

def make_ones(batch_size):
  """Creates a tensor of ground truths of real data for discrimator"""
  return Variable(torch.ones(batch_size, 1)).to(DEVICE)

def make_zeros(batch_size):
  """Creates a tensor of ground truths of fake data for discrimator"""
  return Variable(torch.zeros(batch_size, 1)).to(DEVICE)



def train_discriminator(discriminator, d_opt, loss, real_data, fake_data):
  d_opt.zero_grad()

  p_real = discriminator(real_data)
  loss_real = loss(p_real, make_ones(len(real_data)))
  loss_real.backward()

  p_fake = discriminator(fake_data)
  loss_fake = loss(p_fake, make_zeros(len(fake_data)))
  loss_fake.backward()

  d_opt.step()

  return loss_fake + loss_real

def train_generator(discrimnator, fake_data, g_opt, loss):
  g_opt.zero_grad()

  pred = discrimnator(fake_data)

  g_loss = loss(pred, make_ones(len(fake_data)))

  g_loss.backward()

  g_opt.step()

  return g_loss





def train(dataloader, generator, discrimator, opt_g, opt_d, loss, epochs, k):
  g_loss_list = []
  d_loss_list = []
  test_noise = noise(64)

  generator.train()
  discrimator.train()

  for epoch in range(epochs):
    g_loss_sum = 0.0
    d_loss_sum = 0.0

    for i, data in enumerate(dataloader):
      images, _ = data
      batch_size = len(images)

      for _ in range(k):
        fake_data = generator(noise(batch_size)).detach()
        real_data = images.to(DEVICE)
        d_loss_sum += train_discriminator(discrimator, opt_d, loss, real_data, fake_data)

      fake_data = generator(noise(batch_size))
      g_loss_sum += train_generator(discrimator, fake_data, opt_g, loss)

    images = generator(test_noise).cpu().detach()
    images = make_grid(images)
    if epoch % 50 == 0:
      to_image(images[0]).show()
    g_loss_list.append(g_loss_sum / i)
    d_loss_list.append(d_loss_sum / i)
    print(f'Epoch {epoch},  G Loss {g_loss_sum / i}  D Loss {d_loss_sum / i}')

      
    

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])



data = MNIST('./data/', train=True, transform=transform)
trainloader = DataLoader(data, batch_size=100, shuffle=True)

G = Generator(NUM_NOISE_FEATURES, GEN_OUTPUT_FEATURES).to(DEVICE)
G_OPT = torch.optim.Adam(G.parameters(), lr=2e-4)
D = Discriminator(DIS_INPUT_FEATURES, DIS_OUTPUT_FEATURES).to(DEVICE)
D_OPT = torch.optim.Adam(D.parameters(), lr=2e-4)

LOSS = nn.BCELoss()

train(trainloader, G, D, G_OPT, D_OPT, LOSS, 250, 1)




