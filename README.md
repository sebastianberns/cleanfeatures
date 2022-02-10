# Clean Features

Compute ‘clean’ high-level features for a batch of images with a deep computer vision model.

Currently available models:

- Inception v3

This is a custom implementation that builds on the best practices and partially on the implementation of [GaParmar/clean-fid](https://github.com/GaParmar/clean-fid).

## Usage

Assuming that the repository is available in the working directory.

```python
from cleanfeatures import CleanFeatures  # 1.

cf = CleanFeatures('path/to/inception/snapshot/')  # 2.
features = cf(images)  # 3.
```

1. Import the main class.
2. Create a new instance, optionally providing a directory path. This can be either the place the model snapshot is already saved, or the place it should be downloaded and saved to.
3. Pass a batch of images to compute the corresponding 'clean' features

### CleanFeatures parameters

```python
CleanFeatures(model_path='./models', device=None, log='warning')
```

- ```model_path``` (str or Path object): path to directory where model snapshot is saved or should be saved to. Optional, default: './models'.
- ```device``` (str or torch.device): device which the loaded model will be allocated to. Optional, default: 'cuda' if a GPU is available, otherwise 'cpu'.
- ```log``` (str): logging level (options: 'all', 'debug', 'info', 'warning', 'error', 'critical'), where any option will include all subsequent logging levels. Optional, default: 'warning'.

## Dependencies

- Numpy
- Pytorch 1.8 (does not work with version 1.9 because of some issues with the torchscript Inception model)
- requests
- PIL

## References

Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. [*arXiv preprint arXiv:2104.11222*](http://arxiv.org/abs/2104.11222).
