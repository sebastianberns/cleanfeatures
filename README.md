# Clean Features

Compute ‘clean’ high-level features for a batch of images with a pre-trained embedding model.
This is a custom implementation of best practices recommended by [Parmar et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Parmar_On_Aliased_Resizing_and_Surprising_Subtleties_in_GAN_Evaluation_CVPR_2022_paper.html) and partially builds on code from [GaParmar/clean-fid](https://github.com/GaParmar/clean-fid).

Currently available feature embedding models:

- Inception v3
- CLIP

## Setup

### Dependencies

- torch and torchvision (Pytorch)
- numpy
- requests
- PIL (Pillow)
- clip ([openai/CLIP](https://github.com/openai/CLIP))
  - ftfy
  - regex
  - tqdm

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

- ```model_path``` (str or Path object, optional): path to directory where model checkpoint is saved or should be saved to. Default: './models'.
- ```model``` (str, optional): choice of pre-trained feature extraction model. Options:
  - InceptionV3 (default)
  - CLIP
- ```device``` (str or torch.device, optional): device which the loaded model will be allocated to. Default: 'cuda' if a GPU is available, otherwise 'cpu'.
- ```log``` (str, optional): logging level, where any option will include all lower logging levels. Options:
  - all
  - debug
  - info
  - warning (default)
  - error
  - critical
- ```kwargs``` (dict): optional model-specific arguments. See below.

#### CLIP arguments

- ```clip_model``` (str, optional): choice of pre-trained CLIP model. Options:
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

All methods return a tensor of embedded features [B, F], where F is the number of features.

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

Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. [*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 11410–11420.]( https://openaccess.thecvf.com/content/CVPR2022/html/Parmar_On_Aliased_Resizing_and_Surprising_Subtleties_in_GAN_Evaluation_CVPR_2022_paper.html)
