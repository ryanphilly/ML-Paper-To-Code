import torch
import torch.nn as nn
from torch.nn.modules import module

'''
Paper-to-Code
Image-to-Image Translation with Conditional Adversarial Networks
      https://arxiv.org/pdf/1611.07004.pdf

  Both generator and discriminator layers:
      convolution-BatchNorm-ReLu
'''

def sample_block(type: str, # 'up' or 'down' (case sens)
                 channels_in: int,
                 channels_out: int,
                 kernel_size: int=4,
                 padding: int=1,
                 batch_norm: bool=True,
                 tanh: bool=False):
  modules = list()
  if type == 'up':
    modules.append(nn.ConvTranspose2d(
      channels_in,
      channels_out,
      kernel_size,
      stride=2, padding=padding))
  else:
    modules.append(nn.Conv2d(
      channels_in,
      channels_out,
      kernel_size,
      stride=2, padding=padding))

  if batch_norm:
    modules.append(nn.BatchNorm2d(channels_out))

  if type == 'up':
    modules.append(nn.ReLU() if not tanh else nn.Tanh())
  else:
    modules.append(nn.LeakyReLU(0.2))

  return nn.Sequential(*modules)


class UNet(nn.Module):
  """
  UNet is an encoder-decoder network that utilizes
  skip connections(helps with bottleneck issue)
  """
  def __init__(self, input_channels: int, output_channels: int, generator_feats: int):
    super(UNet, self).__init__()
    # functional encoder (allows skip connections)
    self.encoder = nn.ModuleList([
      sample_block('down', input_channels, generator_feats, batch_norm=False),
      sample_block('down', generator_feats, generator_feats*2),
      sample_block('down', generator_feats*2, generator_feats*4),
      sample_block('down', generator_feats*4, generator_feats*8),

      sample_block('down', generator_feats*8, generator_feats*8),
      sample_block('down', generator_feats*8, generator_feats*8),
      sample_block('down', generator_feats*8, generator_feats*8),
      sample_block('down', generator_feats*8, generator_feats*8)])

    # decoder (mirrors encoder except the
    # input channels after the first layer are doubled due
    # to skip connection concatenation)
    self.decoder = nn.ModuleList([
      sample_block('up', generator_feats*8, generator_feats*8),
      sample_block('up', generator_feats*8*2, generator_feats*8),
      sample_block('up', generator_feats*8*2, generator_feats*8),
      sample_block('up', generator_feats*8*2, generator_feats*8),

      sample_block('up', generator_feats*8*2, generator_feats*4),
      sample_block('up', generator_feats*4*2, generator_feats*2),
      sample_block('up', generator_feats*2*2, generator_feats),
      sample_block('up', generator_feats*2, output_channels, tanh=True)
    ])
    
  def forward(self, x):
    skip_connections = list()
    for i, layer in enumerate(self.encoder):
      x = layer(x)
      if (i != len(self.encoder)-1):
        skip_connections.append(x)

    skip_connections.reverse()
    
    for i, layer in enumerate(self.decoder):
      x = layer(x) if i == 0 \
        else layer(torch.cat((x, skip_connections[i-1]), dim=1))

    return x

x = torch.randn((3, 3, 256, 256)).to('cuda')
gen = UNet(3, 3, 64).to('cuda:0')



    