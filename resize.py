import numpy as np
from PIL import Image
import torch


class Resizer:
    """
    Resizer
        width (int), height (int): dimensions of resized output images
        channels (int): number of channels of input and output images
    """
    def __init__(self, channels=3, width=299, height=299):
        self.channels = channels
        self.width = width
        self.height = height

    __call__ = _handle_input

    """
    Resize a batch, an image or a channel

        input (Pytorch tensor): data matrix with values in range (0, 255)
            [B, C, W, H]: batch of images
            [C, W, H]: single image
            [W, H]: individual channel

    Returns a tensor of the resized input
    """
    def _handle_input(self, input):
        dims = len(input.shape)
        if dims == 4:
            return self.batch_resize(input)
        elif dims == 3:
            return self.image_resize(input)
        elif dims == 2:
            return self.channel_resize(input)
        else:
            raise ValueError(f"Input with {dims} dimensions is not supported")

    """
    Resize a batch of images
        batch (tensor): [B, C, W, H]

    Returns a tensor of the resized batch [B, C, X, Y],
    where X and Y are the resized width and height
    """
    def batch_resize(self, batch):
        device = batch.device
        batch_size = batch.shape[0]

        resized_batch = torch.zeros((batch_size, self.channels, self.width, self.height),
                                    dtype=torch.float32, device=device)
        for idx in range(batch_size):
            resized_batch[idx] = self.image_resize(batch[idx])
        return resized_batch

    """
    Resize a single image
        image (tensor): [C, W, H]

    Returns a tensor of the resized image [C, X, Y],
    where X and Y are the resized width and height
    """
    def image_resize(self, image):
        device = image.device
        channels = image.shape[0]
        assert channels == self.channels

        resized_image = torch.zeros((self.channels, self.width, self.height),
                                    dtype=torch.float32, device=device)
        for idx in range(channels):
            resized_image[idx] = self.channel_resize(image[idx, :, :])
        return resized_image

    """
    Resize a single channel image
        channel (tensor): [W, H]

    Returns a tensor of the resized channel [X, Y],
    where X and Y are the resized width and height
    """
    def channel_resize(self, channel):
        device = channel.device
        channel_np = channel.cpu().numpy().astype(np.float32)  # Convert to nparray on CPU
        img = Image.fromarray(channel_np, mode='F')  # Create Image from 32-bit floating point pixels
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)  # Clean resize
        return torch.tensor(np.asarray(img, dtype=np.float32), device=device)

    def __repr__(self):
        return f"Resizer class, {self.channels} x {self.width} x {self.height}"
