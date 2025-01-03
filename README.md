# Clean Features

Compute ‘clean’ high-level features for a batch of images with a pre-trained embedding model.
This is a custom implementation of best practices recommended by [Parmar et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Parmar_On_Aliased_Resizing_and_Surprising_Subtleties_in_GAN_Evaluation_CVPR_2022_paper.html) and partially builds on code from [GaParmar/clean-fid](https://github.com/GaParmar/clean-fid).

CleanFeatures currently supports the following feature embedding models:

- [CLIP](https://arxiv.org/abs/2103.00020)
- [DreamSim](https://dreamsim-nights.github.io)
- DVAE (DALL-E)
- InceptionV3
- Resnet50

Improvements over other implementations:

- Normalization after resize. The per-channel resize operation can change the range of values of an image tensor, causing it to go outside the normalized range [0, 1]. This implementation re-normalizes the image tensor after resize, restoring the previous range of values.
- Clean features can be saved to disk with the [`save` command](#save). 
- The [CleanFeaturesDataset class](#cleanfeaturesdataset-class) allows to load pre-computed features as a data set. 
- The clean [Resize](#clean-resize-transform) operation can further be used as a Pytorch transform.

## Setup

Include `cleanfeatures` directory in Python path. Currently, this module does not have an installer.

### Dependencies

- torch (Pytorch)
- torchvision >= 0.11.0 (requires `torchvision.models.feature_extraction`)
- numpy
- requests
- PIL (Pillow)
- clip ([openai/CLIP](https://github.com/openai/CLIP))
  - ftfy
  - regex
  - tqdm
- dreamsim ([ssundaram21/dreamsim](https://github.com/ssundaram21/dreamsim))
- attrs

## Usage

Assuming that the repository is available in the working directory or Python path.

```python
from cleanfeatures import CleanFeatures  # 1.

cf = CleanFeatures('path/to/model/checkpoint/')  # 2.
features = cf(images)  # 3.
```

1. Import the main class.
2. Create a new instance, optionally providing a directory path. This can be either the place the model checkpoint is already saved, or the place it should be downloaded and saved to.
3. Pass a data source to compute the corresponding clean features.

### Recommendations

- Device: keep your raw data on CPU and specify the target device (e.g. GPU) in the `CleanFeatures` constructor. Resizing needs to be done on the CPU (via Pillow). Passing the raw data to device before introduces unnecessary back-and-forth synchronization between devices. The exception is the generative model data source, which generates images on device.

### CleanFeatures class

```python
cf = CleanFeatures(model_path='./models', model='InceptionV3', device=None,
                   log='warning', **kwargs)
```

- `model_path` (str or Path object, optional): path to directory where model checkpoint is saved or should be saved to. Default: './models'.
- `model` (str, optional): choice of pre-trained feature extraction model. Options:
  - CLIP
  - DVAE (DALL-E)
  - InceptionV3 (default)
  - Resnet50
- `device` (str or torch.device, optional): device which the loaded model will be allocated to. Default: 'cuda' if a GPU is available, otherwise 'cpu'.
- `log` (str, optional): logging level, where any option will include all lower logging levels. Options:
  - all
  - debug
  - info
  - warning (default)
  - error
  - critical
- `kwargs` (dict): optional model-specific arguments. See below.

#### CLIP model-specific arguments

- `clip_model` (str, optional): choice of pre-trained CLIP model. Options:
  - RN50
  - RN101
  - RN50x4
  - RN50x16
  - RN50x64
  - ViT-B/32
  - ViT-B/16
  - ViT-L/14 (default)
  - ViT-L/14@336px

#### DreamSim model-specific arguments

- `dreamsim_type` (str, optional): choice of pre-trained DreamSim model. Options:
  - ensemble (default; all below models together)
  - dino_vitb16
  - clip_vitb32
  - open_clip_vitb32

### Methods

The class provides three methods to process different types of input: a data tensor, a generative model, or a dataloader.

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

#### compute_features_from_model

Directly samples from a pre-trained generative model. 

```python
rng = torch.Generator(device='cuda')  # Optional random number generator
rng.manual_seed(42)  # Random seed for sampling
cf.compute_features_from_model(model, z_dim=512, num_samples=50_000, batch_size=128, rng=rng)
```

- `model` (Module): Pre-trained generative model.
- `z_dim` (int, optional): Number of model input dimensions. Default: 512.
- `num_samples` (int, optional): Number of samples to generate and process. Default: 50,000.
- `batch_size` (int, optional): Batch size for model sampling. Default: 128.
- `rng` (Generator, optional): Random number generator for seeded sampling.

Alias: `compute_features_from_generator(...)`

#### compute_features_from_dataset

Will request images from a Pytorch dataloader.

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

transform = ToTensor()  # No resize, no normalization, just convert to tensor
dataset = CIFAR10(root='data', transform=transform)  # Example data set

cf.compute_features_from_dataset(dataset, num_samples=50_000, batch_size=128,
                                 num_workers=8, shuffle=False)
```

- `dataset` (Dataset): Instance of Pytorch data set.
- `num_samples` (int, optional): Number of samples to process.
- `batch_size` (int, optional): Batch size for sampling. Default: 128.
- `num_workers` (int, optional): Number of parallel threads. Best practice is to set to the number of CPU threads available. Default: 0.
- `shuffle` (bool, optional): Indicates whether samples will be randomly shuffled or not. Default: False.
- `sampler` (Sampler or Iterable, optional): Sampling strategy, instance of `Sampler` or `Iterable` with `__len__` implemented. If set, `shuffle` has to be None. Default: None

#### save

Save the last computed features to file.

```python
cf.save(path="./save", name="features")
```
In the above example, the features will be stored as `./save/features.pt`.

- `path` (str or Path, optional): path to save directory. Default: './'.
- `name` (str, optional): filename without file extension. Default: 'features'.

### Attributes

#### features and targets

Returns the last computed feature and targets tensors if available, `None` otherwise.

```python
features = cf.features
targets = cf.targets  # Only available if data set provides labels
```

### Resize transform

See [documentation](https://github.com/sebastianberns/cleanfeatures/tree/main/transforms).

### CleanFeaturesDataset class

Extends Pytorch Dataset class to load a pre-computed features data set.
Provides length (1.) and indexing methods (2.) for use with a data loader (3.)

```python
from cleanfeatures import CleanFeaturesDataset
from torch.utils.data import DataLoader

dataset = CleanFeaturesDataset(path, map_location=None)
num_samples = len(dataset)  # 1.
feature = dataset[0]  # 2.

dataloader = DataLoader(dataset, batch_size=128, num_workers=8)  # 3.
```

- `path` (str or Path object): path to feature tensor file.
- `map_location` (str or torch.device, optional): device which the loaded data set will be allocated to. Default: 'None', file will be loaded onto the same device it was saved from.

## References

Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. [*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 11410–11420.](https://openaccess.thecvf.com/content/CVPR2022/html/Parmar_On_Aliased_Resizing_and_Surprising_Subtleties_in_GAN_Evaluation_CVPR_2022_paper.html)
