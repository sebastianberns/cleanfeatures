# Clean Features

Compute ‘clean’ high-level features for a batch of images with a pre-trained model.

Currently available models:

- Inception v3
- CLIP

This is a custom implementation that builds on the best practices and partially on the implementation of [GaParmar/clean-fid](https://github.com/GaParmar/clean-fid).

## Dependencies

- pytorch \*
- numpy
- requests
- Pillow (PIL)
- ftfy
- regex

\* This repository not work with Pytorch version 1.9 because of some issues with the torchscript Inception model.

## Usage

Assuming that the repository is available in the working directory.

```python
from cleanfeatures import CleanFeatures  # 1.

cf = CleanFeatures('path/to/model/checkpoint/')  # 2.
features = cf(images)  # 3.
```

1. Import the main class.
2. Create a new instance, optionally providing a directory path. This can be either the place the model checkpoint is already saved, or the place it should be downloaded and saved to.
3. Pass a batch of images to compute the corresponding 'clean' features

### CleanFeatures class arguments

```python
CleanFeatures(model_path='./models', device=None, log='warning', **kwargs)
```

- ```model_path``` (str or Path object): path to directory where model checkpoint is saved or should be saved to. Optional, default: './models'.
- ```model``` (str): choice of pre-trained feature extraction model. Options:
  - InceptionV3W (default)
  - CLIP
- ```device``` (str or torch.device): device which the loaded model will be allocated to. Optional, default: 'cuda' if a GPU is available, otherwise 'cpu'.
- ```log``` (str): logging level, where any option will include all lower logging levels. Options:
  - all
  - debug
  - info
  - warning (default)
  - error
  - critical
- ```kwargs``` (dict): optional model-specific arguments. See below.

#### CLIP arguments

- ```clip_model``` (str): choice of pre-trained CLIP model. Options:
  - RN50
  - RN101
  - RN50x4
  - RN50x16
  - RN50x64
  - ViT-B/32
  - ViT-B/16
  - ViT-L/14 (default)
  - ViT-L/14@336px

### Methods

The class provides three methods to process different types of input: a data tensor, a generator network, or a dataloader.

All methods return a tensor of features [B, F] in range (-1, +1), where F is the number of features.

#### compute_features

Expects a Pytorch tensor as input, either a batch of images, a single image or an individual channel.

```python
cf.compute_features(input)
```

- `input` (Pytorch tensor): data matrix with values in range (0, 255).
  - Shape [B, C, W, H]: batch of images.
  - Shape [C, W, H]: single image.
  - Shape [W, H]: individual channel.

#### compute_features_from_generator

Directly samples from a pre-trained generator.

```python
cf.compute_features_from_generator(generator, z_dim=512, num_samples=50_000,
                                   batch_size=128)
```

- `generator` (Module): Pre-trained generator model.
- `z_dim` (int): Number of generator input dimensions. Optional, default: 512.
- `num_samples` (int): Number of samples to generate and process. Optional, default: 50,000.
- `batch_size` (int): Batch size for generator sampling. Optional, default: 128.

#### compute_features_from_dataset

Will request images from a Pytorch dataloader.

```python
cf.compute_features_from_dataset(dataloader, num_samples=50_000, batch_size=128)
```

- `dataloader` (DataLoader): Instance of Pytorch data loader.
- `num_samples` (int): Number of samples to process. Optional, default: 50,000.
- `batch_size` (int): Batch size for sampling. Optional, default: 128.

## References

Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. [*arXiv preprint arXiv:2104.11222*](http://arxiv.org/abs/2104.11222).
