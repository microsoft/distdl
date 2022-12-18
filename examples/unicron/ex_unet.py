from distdl_unet import ClassicalUNet
import torch


# Input data
batch_size = 2
in_channels = 3
features = 128
x = torch.randn(batch_size, in_channels, features, features, features, device='cuda:0')

# Unet
levels = 5
base_channels = 32  # Crash: 512; Okay: 256, 128
out_channels = 1

unet = ClassicalUNet(3, levels, in_channels, base_channels, out_channels).to('cuda:0')

y = unet(x)
y.sum().backward()