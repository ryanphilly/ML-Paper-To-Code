
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision import transforms

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

def noise(batch_size, n_features=100, device='cuda'):
  """creates a noise matrix for a given batch size"""
  return Variable(torch.randn(batch_size, n_features)).to(device)

def make_ones(batch_size):
  """Creates a tensor of ground truths of real data for discrimator"""
  return Variable(torch.ones(batch_size, 1)).to('cuda')

def make_zeros(batch_size):
  """Creates a tensor of ground truths of fake data for discrimator"""
  return Variable(torch.zeros(batch_size, 1)).to('cuda')



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
  to_image = transforms.ToPILImage()
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
        real_data = images.to('cuda')
        d_loss_sum += train_discriminator(discrimator, opt_d, loss, real_data, fake_data)

      fake_data = generator(noise(batch_size))
      g_loss_sum += train_generator(discrimator, fake_data, opt_g, loss)

    if epoch % 50 == 0:
      images = generator(test_noise).cpu().detach()
      images = make_grid(images)
      to_image(images[0]).show()

    g_loss_list.append(g_loss_sum / i)
    d_loss_list.append(d_loss_sum / i)
    print(f'Epoch {epoch},  G Loss {g_loss_sum / i}  D Loss {d_loss_sum / i}')


noisee = noise(64)

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])

data = MNIST('./data/', train=True, transform=transform)
trainloader = DataLoader(data, batch_size=100, shuffle=True)


generator = Gen(noisee.shape[1], 1).to('cuda')
discriminator = Dis().to('cuda')
generator.apply(weights_init)
discriminator.apply(weights_init)

opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

loss = nn.BCELoss()

train(trainloader, generator, discriminator, opt_g, opt_d, loss, 250, 1)