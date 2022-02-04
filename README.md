# Clean Features

Compute ‘clean’ high-level features for a batch of images with a deep computer vision model.

Currently available models:

- Inception v3

This is a custom implementation that builds on the best practices and partially on the implementation of [GaParmar/clean-fid](https://github.com/GaParmar/clean-fid).

## Usage

```python
from cleanfeatures import CleanFeatures  # 2.

cf = CleanFeatures('path/to/inception/snapshot/')  # 3.
features = cf(images)  # 4.
```

1. Copy the repository to the working directory.
2. Import the main class.
3. Create a new instance, optionally providing a directory path. This can be either the place the model snapshot is already saved, or the place it should be downloaded and saved to.
4. Pass a batch of images to compute the corresponding 'clean' features

## References

Parmar, G., Zhang, R., & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. [*arXiv preprint arXiv:2104.11222*](http://arxiv.org/abs/2104.11222).
