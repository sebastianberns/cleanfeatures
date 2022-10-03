import numpy as np
from PIL import Image
import torch


if not hasattr(Image, 'Resampling'):  # Pillow < 9.1
    # Constants deprecated since version 9.1.0
    # Replaced by enum.IntEnum class
    Image.Resampling = Image


class Resize:
    """
    Resize
        channels (int): number of channels of input and output images
        width (int), height (int): dimensions of resized output images
        filter (PIL.Image.Resampling): resampling filters. Default (and recommended): BICUBIC
        normalize (bool, optional): whether to change the range of values to
            the original values after resize. Default: True
    """
    def __init__(self, channels=3, width=299, height=299, filter=Image.Resampling.BICUBIC, normalize=True):
        self.channels = channels
        self.width = width
        self.height = height
        self.filter = filter
        self.normalize = normalize

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

    __call__ = _handle_input

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
        for i in range(batch_size):
            resized_batch[i] = self.image_resize(batch[i])
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
        vmin, vmax = image.min(), image.max()  # Original min and max values

        resized_image = torch.zeros((channels, self.width, self.height),
                                    dtype=torch.float32, device=device)
        for c in range(channels):
            resized_image[c] = self.channel_resize(image[c, :, :])


        if self.normalize:
            resized_image = self._normalize_after_resize(resized_image, vmin, vmax)
        resized_image = self._augment_channels(resized_image)
        return resized_image

    """
    Normalize image values after resize to previous value range
    Helper function for image_resize()

        x (Tensor): image with values in current range [C, W, H]
        tmin, tmax (float): min and max values of target range (original image values)

    Returns image tensor normalized to original value range [C, W, H]
    """
    def _normalize_after_resize(self, x, tmin, tmax):
        # vmin, vmax : target min and max values (original)
        cmin, cmax = x.min(), x.max()  # Current min and max values (resized)

        y = x - cmin           # current values - current min
        y = y * (tmax - tmin)  # * target range
        y = y / (cmax - cmin)  # / current range
        y = y + tmin           # + target min
        return y

    """
    Augment number of channels to meet the image requirements
    Helper function for image_resize()

        input (Tensor): image of variable number of channels [X, W, H]

    Returns an image with augmented channels [C, W, H]
    """
    def _augment_channels(self, image):
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
    Resize a single channel image
        channel (tensor): [W, H]

    Returns a tensor of the resized channel [X, Y],
    where X and Y are the resized width and height
    """
    def channel_resize(self, channel):
        device = channel.device
        channel_np = channel.cpu().numpy().astype(np.float32)  # Convert to nparray on CPU
        img = Image.fromarray(channel_np, mode='F')  # Create Image from 32-bit floating point pixels
        img = img.resize((self.width, self.height), resample=self.filter)  # Clean resize
        return torch.tensor(np.asarray(img, dtype=np.float32), device=device)

    def __repr__(self):
        return f"Resize, {self.channels} x {self.width} x {self.height} [C, W, H]"
