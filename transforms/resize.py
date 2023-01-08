from enum import IntEnum
from typing import Optional, Union

import numpy as np
from PIL import Image  # type: ignore[import]
from PIL.Image import Image as PILImage  # type: ignore[import]
import torch
from torch import Tensor
import torchvision.transforms.functional as F  # type: ignore[import]


if not hasattr(Image, 'Resampling'):  # Pillow < 9.1
    # Constants deprecated since version 9.1.0
    # Replaced by enum.IntEnum class
    Image.Resampling = Image


class Resize:
    """
    Resize
        width (int): dimensions of resized output images
        height (int, optional): height of resized output images, defaults to width (square image)
        channels (int, optional): number of channels of input and output images. Default: None (no channel augmentation)
        filter (PIL.Image.Resampling): resampling filters. Default (and recommended): BICUBIC
        normalize (bool, optional): whether to change the range of values to
            the original values after resize. Default: True
    """
    def __init__(self, width: int, height: Optional[int] = None, channels: Optional[int] = None, 
                 filter: Union[int, IntEnum] = Image.Resampling.BICUBIC, normalize: bool = True) -> None:
        self.width = width
        self.height = width if height is None else height
        self.channels = channels
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
        batch (tensor): [B, C, H, W]
    Return a tensor of the resized batch [B, C, Y, X],
    where X and Y are the resized width and height
    """
    def tensor_batch_resize(self, batch: Tensor) -> Tensor:
        device = batch.device
        batch_size = batch.shape[0]
        num_channels = batch.shape[1] if not self.channels else self.channels

        resized_batch = torch.zeros((batch_size, num_channels, self.height, self.width),
                                    dtype=torch.float32, device=device)
        for i in range(batch_size):
            resized_batch[i] = self.tensor_instance_resize(batch[i])
        return resized_batch

    """
    Resize a single image
        image (tensor): [C, H, W]
    Return a tensor of the resized image [C, Y, X],
    where X and Y are the resized width and height
    """
    def tensor_instance_resize(self, image: Tensor) -> Tensor:
        device: torch.device = image.device
        channels: int = image.shape[0]

        resized_image = torch.zeros((channels, self.height, self.width),
                                    dtype=torch.float32, device=device)
        for c in range(channels):
            channel = image[c, :, :]
            vmin, vmax = channel.min(), channel.max()  # Channel min and max values
            resized_channel = self.tensor_channel_resize(channel)
            if self.normalize:
                resized_channel = self._normalize_channel_after_resize(resized_channel, vmin, vmax)
            print(resized_image[c].shape, resized_channel.shape, 'tensor_instance_resize')
            resized_image[c] = resized_channel
        
        resized_image = self._match_channels(resized_image)
        return resized_image

    """
    Normalize channel values after resize to previous range
    Helper function for tensor_instance_resize()
        x (Tensor): image channel with values in current range [H, W]
        tmin, tmax (Tensor): min and max values of target range (original image values)
    Return image channel tensor normalized to original value range [H, W]
    """
    def _normalize_channel_after_resize(self, x: Tensor, tmin: Tensor, tmax: Tensor) -> Tensor:
        # tmin, tmax : target min and max values (original)
        cmin, cmax = x.min(), x.max()  # Current min and max values (resized)
        trange = (tmax - tmin)  # Target range
        crange = (cmax - cmin)  # Current range

        if trange > 0 and crange > 0:  # Prevent division by zero
            x = x - cmin    # Subtract current minimum value
            x = x * trange  # Multiply by target range
            x = x / crange  # Divide by current range
            x = x + tmin    # Add target minimum value
        return x

    """
    Adjust number of channels to meet the image requirements
    Helper function for tensor_instance_resize()
        input (Tensor): image of variable number of channels [X, H, W]
    Return an image with adjusted number of channels [C, H, W]
    """
    def _match_channels(self, tensor: Tensor) -> Tensor:
        device = tensor.device
        channels = tensor.shape[0]

        if self.channels is not None and self.channels != channels:
            if channels == 1 and self.channels == 3:  # L to RGB
                mode_in = 'L'
                mode_out = 'RGB'
            if channels == 3 and self.channels == 1:  # RGB to L
                mode_in = 'RGB'
                mode_out = 'L'
            else:
                raise NotImplementedError(f"Currently no strategy to adjust "
                                          f"images of {channels} channels to "
                                          f"{self.channels} channels")
            
            image = F.to_pil_image(tensor, mode=mode_in)  # PIL Image [W, H, C]
            image = image.convert(mode=mode_out)  # Convert to target mode
            tensor = F.to_tensor(image).to(device)  # Tensor [C, H, W]
        return tensor

    """
    Resize a single channel tensor
        channel (tensor): [H, W]
    Return a tensor of the resized channel [Y, X],
    where X and Y are the resized width and height
    """
    def tensor_channel_resize(self, channel: Tensor) -> Tensor:
        device = channel.device
        im = F.to_pil_image(channel, mode='F')  # PIL Image, 32-bit floating point [W, H]
        im_r = self.image_channel_resize(im)  # Clean resize
        ch_r = F.to_tensor(im_r).to(device)  # Tensor, float32 [H, W]
        return ch_r

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
        channel (Image): PIL Image, mode 'F' [W, H]
    Return a resized PIL Image
    """
    def image_channel_resize(self, channel: PILImage) -> PILImage:
        resized_channel = channel.resize((self.width, self.height), resample=self.filter)
        print(resized_channel.size, resized_channel.getbands(), 'image_channel_resize')
        return resized_channel

    def __repr__(self) -> str:
        return f"Resize(width={self.width}, height={self.height}, channels={self.channels})"
