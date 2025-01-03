#!/usr/bin/env python

import logging
from pathlib import Path
from typing import Tuple, Union, Optional, Iterable

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler

from . import models
from .transforms import Resize


"""
Standard Python logging levels
https://docs.python.org/3/library/logging.html#levels

Higher levels (lower is value) include all lower levels.
E.g. 'critical' only shows critical messages, whereas 'warning' will show
warnings, errors and critical messages.
"""
log_levels = {
    'all': 0,  # Log all messages
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 40,  # Log both errors and critical messages
    'critical': 50  # Only log critical messages
}


"""
CleanFeatures class

Attributes
    model (torch.nn.Module): Embedding model
    num_features (int): Number of dimensions of feature output
    features (None, tensor): Computed features of shape [B, F]

Methods
    self: Compute features from any type of input
    compute_features: Direct access to processing pipeline
    compute_features_from_model: Sample image-generating model
    compute_features_from_dataset: Get sample from dataset
    save: Save computed feature tensor to path
    set_log_level: Set level of logging output
"""
class CleanFeatures:
    """
    Initialize clean features processor

        model_path (str or Path): Path to state dict of embedding model
        model (str): Name of embedding model
        device (str or device): Name of device for execution
        log (str): Log level
        kwargs (dict): Additional parameters for embedding model
    """
    def __init__(
        self, 
        model_path: Union[str, Path] = './models', 
        model: str = 'InceptionV3', 
        device: Optional[Union[str, torch.device]] = None, 
        log: str = 'warning', 
        **kwargs
    ) -> None:

        # Check if model is implemented
        assert hasattr(models, model), f"Model '{model}' is not available"

        model_path = self._get_path(model_path)

        # If device == None, set to cuda if available, otherwise set to cpu
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.set_log_level(log)

        logging.info('Loading model')
        model_fn = getattr(models, model)  # Get callable from string
        self.model = model_fn(path=model_path, device=self.device, **kwargs)  # Load embedding model to device

        logging.info('Building resize')
        self.resize = Resize(width=self.model.input_width,
                             height=self.model.input_height,
                             channels=self.model.input_channels)
        self.num_features = self.model.num_features
        self.dtype = self.model.dtype

        self._features: Optional[Tensor] = None
        self._targets: Optional[Tensor] = None  # Only set when processing a data set with labels

        logging.info('CleanFeatures ready')

    @property
    def features(self) -> Optional[Tensor]:
        return self._features

    @property
    def targets(self) -> Optional[Tensor]:
        return self._targets

    """
    Redirect calls based on input data type

        input   Tensor: directly process input
                Module: sample batch from image-generating model
                Dataset: load batch from data set
    """
    def _handle_input(
        self, 
        input: Union[Tensor, nn.Module, Dataset], 
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        if isinstance(input, Tensor):  # Tensor ready for processing
            return self.compute_features_from_samples(input, **kwargs)
        elif isinstance(input, nn.Module):  # Image-generating model
            return self.compute_features_from_model(input, **kwargs)
        elif isinstance(input, Dataset):  # Data set
            return self.compute_features_from_dataset(input, **kwargs)
        else:
            raise ValueError(f"Input type {type(input)} is not supported")

    """
    Direct call to CleanFeatures instance

        ```python
        cf = CleanFeatures(path)
        feature = cf(input)
        ```
    """
    __call__ = _handle_input

    """
    Perform a gradient-free model forward pass on device (GPU if available)
    """
    def _model_fwd(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(input.to(self.device))

    """
    Augment data dimensions to make it meet the model input requirements

        input (Tensor): data of variable number of dimensions

    Returns a tensor of the input data [B, 3, W, H]
    """
    def _augment_dimensions(self, input: Tensor) -> Tensor:
        dims = len(input.shape)

        # Adjust number of dimensions
        if dims == 2:  # [W, H]
            input.unsqueeze_(0)  # Add channel dimension
        if dims == 3:  # [C, W, H]
            input.unsqueeze_(0)  # Add batch dimension

        # Now all input is standardized to [B, C, W, H]
        return input

    """
    Compute features of small batch, single image or individual channel
    Convenient for batches of samples small enough to be processed in one go

        input (Pytorch tensor): data matrix with values in range (0, 255)
            [B, C, W, H]: small batch of images
            [C, W, H]: single image
            [W, H]: individual channel

    Returns a tensor of features [B, F], where F is the number of features
    """
    def compute_features(self, input: Tensor) -> Tensor:
        input = self.resize(input)  # Clean resize on CPU
        input = self._augment_dimensions(input)  # Adjust input dimensions
        features = self._model_fwd(input)  # Embed model forward pass (on GPU if available)
        features = features.to(self.dtype)  # Convert to data type
        return features

    """
    Compute features from a tensor samples 
    Conventient when number of samples is too big to be parsed in one go

        samples (Tensor): Matrix of image samples [B, C, W, H]
        batch_size (int, optional): Batch size for sample processing. Default: 128

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_samples(
        self, 
        samples: Tensor, 
        batch_size: int = 128
    ) -> Tensor:
        num_samples = samples.shape[0]  # Number of samples
        logging.info(f"Computing features for {num_samples:,} samples")
        features = torch.zeros((num_samples, self.num_features),
                               dtype=self.dtype, device=self.device)  # Allocate memory on device
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            batch_samples = samples[c:c+b]
            features[c:c+b] = self.compute_features(batch_samples)
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        logging.info("Computed features for {0:,} samples in {1} dimensions.".format(*features.shape))
        return features

    """
    Compute features of samples from pre-trained image-generating model
    Expected output of the model is a tensor of size [B, C, W, H],
    where B is the batch size equal to the input

        model (Module): Pre-trained image-generating model
        z_dim (int): Number of model input dimensions
        num_samples (int): Number of samples to generate and process
        batch_size (int, optional): Batch size for model sampling. Default: 128

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_model(
        self,
        model: nn.Module,
        z_dim: int,
        num_samples: int,
        batch_size: int = 128,
        rng: Optional[torch.Generator] = None
    ) -> Tensor:
        logging.info(f"Computing features for {num_samples:,} samples from model")
        model.eval()
        features = torch.zeros((num_samples, self.num_features),
                               dtype=self.dtype, device=self.device)  # Allocate memory on device
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            z = torch.randn((b, z_dim), generator=rng, device=self.device)  # Random samples
            with torch.no_grad():
                samples = model(z)  # Generate images

            features[c:c+b] = self.compute_features(samples)  # Compute and append
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        logging.info("Computed features for {0:,} batch items in {1} dimensions.".format(*features.shape))
        return features

    # Alias for backward compatibility
    compute_features_from_generator = compute_features_from_model

    """
    Compute features of samples from data set
    Assumes data samples are transformed to Tensor in range [0, 1]

        dataset (Dataset): Instance of Pytorch data set
        num_samples (int): Number of samples to process
        batch_size (int, optional): Batch size for sampling. Default: 128
        num_workers (int, optional): Number of parallel threads. Best practice
            is to set to the number of CPU threads available. Default: 0
        shuffle (bool, optional): Indicates whether samples will be randomly
            shuffled or not. Default: False
        sampler (Sampler or Iterable, optional): Sampling strategy, instance of 
            Sampler or Iterable with `__len__` implemented. If set, `shuffle` 
            has to be None. Default: None

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_dataset(
        self, 
        dataset: Dataset, 
        num_samples: int, 
        batch_size: int = 128, 
        num_workers: int = 0, shuffle: Optional[bool] = None, 
        sampler: Optional[Union[Sampler, Iterable]] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        logging.info(f"Computing features for {num_samples:,} samples from data set")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                sampler=sampler, num_workers=num_workers)
        dataiterator = iter(dataloader)
        features = torch.zeros((num_samples, self.num_features),
                               dtype=self.dtype, device=self.device)  # Allocate memory on device

        targets = None  # Default: do not process targets
        if hasattr(dataset, 'targets'):  # Unless dataset provides targets
            # Determine targets dimensionality and data type
            if isinstance(dataset.targets, list):  # List
                targets_shape: Tuple[int, ...] = (num_samples, )  # Assuming one dimension
                targets_dtype = torch.int64  # Default int data type
            elif type(dataset.targets) in [np.ndarray, torch.Tensor]:  # Numpy array or Tensor
                _, *target_dims = dataset.targets.shape  # Possibly multiple dimensions
                targets_shape: Tuple[int, ...] = (num_samples, *target_dims)  # type: ignore[no-redef]
                targets_dtype = dataset.targets.dtype
            else:  # Other data types not implemented
                raise NotImplementedError(f"Data set targets of type '{type(dataset.targets)}' currently not supported")
            targets = torch.zeros(targets_shape, dtype=targets_dtype, device=self.device)

        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            samples, labels = next(dataiterator)  # Load samples and labels
            samples = samples[:b].to(self.device)  # Limit to batch size and pass to device
            labels = labels[:b].to(self.device)

            features[c:c+b] = self.compute_features(samples)  # Compute and append
            if targets is not None:
                targets[c:c+b] = labels  # Collect target labels

            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        self._targets = targets
        logging.info("Computed features for {0:,} batch items in {1} dimensions.".format(*features.shape))
        return features, targets

    def save(
        self, 
        path: Union[str, Path] = "./", 
        name: str = "features"
    ) -> None:
        dir = self._get_path(path)
        dir.mkdir(exist_ok=True)  # Create save directory
        save_path = dir/f"{name}.pt"

        data = {'features': self.features, 'targets': self.targets}
        torch.save(data, save_path)

        logging.info(f"Features and targets saved to '{save_path}'")

    def set_log_level(self, log_level: str) -> None:
        assert log_level in log_levels.keys(), f"Log level {log_level} not available"
        logging.basicConfig(format='%(message)s', level=log_levels[log_level])

    def _get_path(self, path: Union[str, Path]) -> Path:
        # Ensure clean absolute Path object
        return Path(path).expanduser().resolve()

    def __repr__(self) -> str:
        return f"CleanFeatures, {self.model.name} embedding, {self.num_features} features"
