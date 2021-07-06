import torch
import torch.nn as nn

'''
Paper-to-Code
Image-to-Image Translation with Conditional Adversarial Networks
      https://arxiv.org/pdf/1611.07004.pdf

  Both generator and discriminator layers:
      convolution-BatchNorm-ReLu
'''

def downsample_block(channels_in: int, channels_out: int, kernel_size: int=4) -> nn.Sequential:
  '''Single layer of the UNet encoder'''
  return nn.Sequential(
    nn.Conv2d(channels_in, channels_out, kernel_size, stride=2),
    nn.BatchNorm2d(channels_out),
    nn.LeakyReLU(0.2))

def upsample_block(channels_in: int, channels_out: int, kernel_size: int=4) -> nn.Sequential:
  '''Single layer of the UNet decoder'''
  return nn.Sequential(
    nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride=2),
    nn.BatchNorm2d(channels_out),
    nn.ReLU())


class UNet(nn.Module):
  """
  UNet is an encoder-decoder network that utilizes
  skip connections(helps with bottleneck issue)
  """
  def __init__(self, input_channels: int, output_channels: int, generator_feats: int):
    super(UNet, self).__init__()
    # functional encoder (allows skip connections)
    self.encoder = nn.ModuleList([
      downsample_block(input_channels, generator_feats),
      downsample_block(generator_feats, generator_feats*2),
      downsample_block(generator_feats*2, generator_feats*4),
      downsample_block(generator_feats*4, generator_feats*8),

      downsample_block(generator_feats*8, generator_feats*8),
      downsample_block(generator_feats*8, generator_feats*8),
      downsample_block(generator_feats*8, generator_feats*8),
      downsample_block(generator_feats*8, generator_feats*8)])

    # decoder (mirrors encoder except the
    # input channels after first layer are doubled due
    # to skip connection concatenation)
    self.decoder = nn.ModuleList([
      upsample_block(generator_feats*8, generator_feats*8),
      upsample_block(generator_feats*8*2, generator_feats*8),
      upsample_block(generator_feats*8*2, generator_feats*8),
      upsample_block(generator_feats*8*2, generator_feats*8),

      upsample_block(generator_feats*8*2, generator_feats*4),
      upsample_block(generator_feats*4*2, generator_feats*2),
      upsample_block(generator_feats*2*2, generator_feats),
      upsample_block(generator_feats*2, output_channels)
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
              else layer(torch.cat(x, skip_connections[i-1]))

    return x


gen = UNet(1, 3).to('cuda:0')


    