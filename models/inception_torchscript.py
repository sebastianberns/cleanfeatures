from pathlib import Path

import torch
from torch.nn import Module

from .helpers import check_download_url


class InceptionV3W(Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

        path (str): locally saved inception model snapshot
    """
    def __init__(self, path='./models', device=None):
        super(InceptionV3W, self).__init__()

        path = Path(path)  # Make sure this is a Path object

        self.name = "InceptionV3"
        self.url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
        self.input_channels = 3
        self.input_width = 299
        self.input_height = 299
        self.num_features = 2048

        # download the network if it is not present at the given directory
        self.path = path = check_download_url(path, self.url)
        self.device = device

        self.base = torch.jit.load(path, map_location=device).eval()
        self.embedding = self.base.layers


    """
    Get the inception features without resizing

        x (Pytorch tensor [B, C, W, H]): Image with values in range (0, 255)

    Returns a Pytorch tensor [B, F] in range (-1, +1),
    where F is the number of features
    """
    def forward(self, x):
        B, C, W, H = x.shape  # Batch size, channels, width, height

        # Make sure input matches expected dimensions
        assert (W == self.input_width) and (H == self.input_height)

        # Normalization
        # Change value range from (0, 255) to (-1, +1)
        x = x - 128.  # Center around zero
        x = x / 128.  # Scale to (-1, +1)

        features = self.embedding.forward(x, ).view((B, self.num_features))
        return features
