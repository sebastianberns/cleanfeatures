from enum import IntEnum
from typing import Optional, Union

import numpy as np
from PIL import Image  # type: ignore[import]
from PIL.Image import Image as PILImage  # type: ignore[import]
import torch
from torch import Tensor


if not hasattr(Image, 'Resampling'):  # Pillow < 9.1
    # Constants deprecated since version 9.1.0
    # Replaced by enum.IntEnum class
    Image.Resampling = Image


class Resize:
    """
    Resize
        width (int): dimensions of resized output images
        height (int, optional): height of resized output images, defaults to width (square image)
        channels (int): number of channels of input and output images. Default: 3 (RGB)
        filter (PIL.Image.Resampling): resampling filters. Default (and recommended): BICUBIC
        normalize (bool, optional): whether to change the range of values to
            the original values after resize. Default: True
    """
    def __init__(self, width: int, height: Optional[int] = None, channels: int = 3, 
                 filter: Union[int, IntEnum] = Image.Resampling.BICUBIC, normalize: bool = True) -> None:
        self.channels = channels
        self.width = width
        self.height = width if height is None else height
        self.filter = filter
        self.normalize = normalize

    """
    Handle input and send to corresponding methods based on type
        input (Image): image object with values in range [0, 254]
        input (Tensor): data matrix with values in range [0, 1]
    Return the resized input of the same type as input (Image or Tensor)
    """
    def _handle_input(self, input: Union[PILImage, Tensor]) -> Union[PILImage, Tensor]:
        
        if isinstance(input, PILImage):
            return self.image_resize(input)
        elif isinstance(input, Tensor):
            return self._handle_tensor_input(input)
        else:
            raise TypeError("Resize only supports PIL Image and Tensor input.")

    __call__ = _handle_input

    """
    Handle Tensor input and send to corresponding methods based on number of dimensions
        input (Tensor): data matrix with values in range [0, 1]
            [B, C, W, H]: batch of instances
            [C, W, H]: single instance
            [W, H]: individual channel
    Return a tensor of the resized input
    """
    def _handle_tensor_input(self, input: Tensor) -> Tensor:
        num_dims = len(input.shape)
        if num_dims == 4:
            return self.tensor_batch_resize(input)
        elif num_dims == 3:
            return self.tensor_instance_resize(input)
        elif num_dims == 2:
            return self.tensor_channel_resize(input)
        else:
            raise ValueError(f"Input with {num_dims} dimensions is not supported")

    """
    Resize a batch of images
        batch (tensor): [B, C, W, H]
    Return a tensor of the resized batch [B, C, X, Y],
    where X and Y are the resized width and height
    """
    def tensor_batch_resize(self, batch: Tensor) -> Tensor:
        device = batch.device
        batch_size = batch.shape[0]

        resized_batch = torch.zeros((batch_size, self.channels, self.width, self.height),
                                    dtype=torch.float32, device=device)
        for i in range(batch_size):
            resized_batch[i] = self.tensor_instance_resize(batch[i])
        return resized_batch

    """
    Resize a single image
        image (tensor): [C, W, H]
    Return a tensor of the resized image [C, X, Y],
    where X and Y are the resized width and height
    """
    def tensor_instance_resize(self, image: Tensor) -> Tensor:
        device: torch.device = image.device
        channels: int = image.shape[0]

        resized_image = torch.zeros((channels, self.width, self.height),
                                    dtype=torch.float32, device=device)
        for c in range(channels):
            channel = image[c, :, :]
            vmin, vmax = channel.min(), channel.max()  # Channel min and max values
            resized_channel = self.tensor_channel_resize(channel)
            if self.normalize:
                resized_channel = self._normalize_channel_after_resize(resized_channel, vmin, vmax)
            resized_image[c] = resized_channel
        
        resized_image = self._augment_channels(resized_image)
        return resized_image

    """
    Normalize image values after resize to previous value range
    Helper function for tensor_instance_resize()
        x (Tensor): image with values in current range [C, W, H]
        tmin, tmax (Tensor): min and max values of target range (original image values)
    Return image tensor normalized to original value range [C, W, H]
    """
    def _normalize_channel_after_resize(self, x: Tensor, tmin: Tensor, tmax: Tensor) -> Tensor:
        # tmin, tmax : target min and max values (original)
        cmin, cmax = x.min(), x.max()  # Current min and max values (resized)

        y = x - cmin           # Subtract current minimum value
        y = y * (tmax - tmin)  # Multiply by target range
        y = y / (cmax - cmin)  # Divide by current range
        y = y + tmin           # Add target minimum value
        return y

    """
    Augment number of channels to meet the image requirements
    Helper function for tensor_instance_resize()
        input (Tensor): image of variable number of channels [X, W, H]
    Return an image with augmented channels [C, W, H]
    """
    def _augment_channels(self, image: Tensor) -> Tensor:
        channels = image.shape[0]

        if channels < self.channels:
            if channels == 1:  # Grayscale image
                # Increase channel dimension with same data (just view, no copy)
                image = image.expand(self.channels, -1, -1)
            else:
                raise NotImplementedError(f"Currently no strategy to augment "
                                          f"images of {channels} channels to "
                                          f"{self.channels} channels")
        return image

    """
    Resize a single channel tensor
        channel (tensor): [W, H]
    Return a tensor of the resized channel [X, Y],
    where X and Y are the resized width and height
    """
    def tensor_channel_resize(self, channel: Tensor) -> Tensor:
        device = channel.device
        channel_np = channel.cpu().numpy().astype(np.float32)  # Convert to nparray on CPU
        img = Image.fromarray(channel_np, mode='F')  # Create image from 32-bit floating point pixels
        img = self.image_channel_resize(img)  # Clean resize
        return torch.tensor(np.asarray(img, dtype=np.float32), device=device)

    """
    Resize PIL Image
        input (Image): image object with values in range [0, 254]
    Return resized image
    """
    def image_resize(self, input: PILImage) -> PILImage:
        resized_channels = []
        for channel in input.split():
            resized_channel = self.image_channel_resize(channel)
            resized_channels.append(resized_channel)
        return Image.merge(input.mode, resized_channels)

    """
    Resize a single image channel
        channel (Image): image in mode L [W, H]
    Return an image of the resized channel [X, Y],
    where X and Y are the resized width and height
    """
    def image_channel_resize(self, channel: PILImage) -> PILImage:
        return channel.resize((self.width, self.height), resample=self.filter)

    def __repr__(self) -> str:
        return f"Resize, {self.channels} x {self.width} x {self.height} [C, W, H]"
