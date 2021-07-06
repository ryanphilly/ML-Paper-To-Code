import torch
from torchvision import transforms
from torch.autograd.variable import Variable
from torchvision.utils import make_grid

def noise(batch_size, n_features, device='cuda'):
  """creates a noise matrix for a given batch size"""
  return Variable(torch.randn(batch_size, n_features)).to(device)

def make_ones(batch_size, device='cuda'):
  """Creates a tensor of ground truths of real data for discrimator"""
  return Variable(torch.ones(batch_size, 1)).to(device)

def make_zeros(batch_size, device='cuda'):
  """Creates a tensor of ground truths of fake data for discrimator"""
  return Variable(torch.zeros(batch_size, 1)).to(device)


tanh_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])

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

def train(dataloader,
          generator,
          discrimator, 
          opt_g,
          opt_d, 
          loss, 
          epochs, 
          k,
          device):

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
        real_data = images.to(device)
        d_loss_sum += train_discriminator(discrimator, opt_d, loss, real_data, fake_data)

      fake_data = generator(noise(batch_size))
      g_loss_sum += train_generator(discrimator, fake_data, opt_g, loss)

    images = generator(test_noise).cpu().detach()
    images = make_grid(images)
    if epoch % 50 == 0:
      transforms.ToPILImage(images[0]).show()
    g_loss_list.append(g_loss_sum / i)
    d_loss_list.append(d_loss_sum / i)
    print(f'Epoch {epoch},  G Loss {g_loss_sum / i}  D Loss {d_loss_sum / i}')
