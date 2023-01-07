# Resize transform

Anti-aliased resizing of PIL Images and Tensors. Can be used in a dataset transform (see example below), before or after conversion to Tensor. Cleanfeatures uses this transform to make image tensors match the embedding model input requirements.

```python
from cleanfeatures import Resize
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor

# After conversion to Tensor
transform = Compose([
    ToTensor(),  # No resize, no normalization, just convert to tensor
    Resize(64)  # Resize Tensor to 64 x 64 [W, H]
])
dataset = CIFAR10(root='data', transform=transform)  # Example data set

# Before conversion to Tensor
transform = Compose([
    Resize(300, 200),  # Resize PIL Image to 300 x 200 px
    ToTensor()
])
```

The Resize class takes the following arguments:

- `width` (int): dimensions of resized output images
- `height` (int, optional): height of resized output images, defaults to same value as width (square image)
- `channels` (int, optional): if set input will be augmented to given number of channels. E.g. argument is set to `channels=3` and input has 1 channel, the input will be expanded to 3 channels. Default: None (no channel augmentation)
- `filter` (PIL.Image.Resampling): resampling filters. Default (and recommended): BICUBIC
- `normalize` (bool, optional): whether to change the range of values to the original values after resize. Default: True
