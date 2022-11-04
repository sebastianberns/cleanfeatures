#!/usr/bin/env python

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

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
    compute_features_from_generator: Sample generator model
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
    def __init__(self, model_path='./models', model='InceptionV3',
                 device=None, log='warning', **kwargs):

        # Check if model is implemented
        assert hasattr(models, model), f"Model '{model}' is not available"

        model_path = self._get_path(model_path)

        # If device == None, set to cuda if available, otherwise set to cpu
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.set_log_level(log)

        logging.info('Loading model')
        model_fn = getattr(models, model)  # Get callable from string
        self.model = model_fn(path=model_path, device=self.device, **kwargs)

        logging.info('Building resize')
        self.resize = Resize(channels=self.model.input_channels,
                             width=self.model.input_width,
                             height=self.model.input_height)
        self.num_features = self.model.num_features
        self.dtype = self.model.dtype

        self._features = None
        self._targets = None  # Only set when processing a data set with labels

        logging.info('CleanFeatures ready')

    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    """
    Redirect calls based on input data type

        input   Tensor: directly process input
                Module: sample batch from generator model
                Dataset: load batch from data set
    """
    def _handle_input(self, input, *kwargs):
        if isinstance(input, torch.Tensor):  # Tensor ready for processing
            return self.compute_features(input)
        elif isinstance(input, nn.Module):  # Generator model
            return self.compute_features_from_generator(input, *kwargs)
        elif isinstance(input, Dataset):  # Data set
            return self.compute_features_from_dataset(input, *kwargs)
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
    Perform a gradient-free model forward pass
    """
    def _model_fwd(self, input):
        with torch.no_grad():
            return self.model(input)

    """
    Augment data dimensions to make it meet the model input requirements

        input (Tensor): data of variable number of dimensions

    Returns a tensor of the input data [B, 3, W, H]
    """
    def _augment_dimensions(self, input):
        dims = len(input.shape)

        # Adjust number of dimensions
        if dims == 2:  # [W, H]
            input.unsqueeze_(0)  # Add channel dimension
            logging.info("Added channel dimension")
        if dims == 3:  # [C, W, H]
            input.unsqueeze_(0)  # Add batch dimension
            logging.info("Added batch dimension")

        # Now all input is standardized to [B, C, W, H]
        return input

    """
    Compute features of batch, image or channel

        input (Pytorch tensor): data matrix with values in range (0, 255)
            [B, C, W, H]: batch of images
            [C, W, H]: single image
            [W, H]: individual channel

    Returns a tensor of features [B, F], where F is the number of features
    """
    def compute_features(self, input):
        logging.info("Resizing ...")
        input = self.resize(input)  # Clean resize

        logging.info("Adjusting number of dimensions ...")
        input = self._augment_dimensions(input)  # Adjust input dimensions

        logging.info("Model forward pass ...")
        features = self._model_fwd(input)  # Embed model forward pass

        features = features.to(self.dtype)  # Convert to data type
        self._features = features
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    """
    Compute features of samples from pre-trained generator

        generator (Module): Pre-trained generator model
        z_dim (int): Number of generator input dimensions
        num_samples (int): Number of samples to generate and process
        batch_size (int, optional): Batch size for generator sampling. Default: 128

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_generator(self, generator, z_dim, num_samples,
                                        batch_size=128):
        logging.info(f"Computing features for {num_samples:,} samples from generator")
        generator.eval()
        features = torch.zeros((num_samples, self.num_features),
                               dtype=self.dtype, device=self.device)
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            z = torch.randn((b, z_dim), device=self.device)  # Random samples
            with torch.no_grad():
                samples = generator(z)  # Generate images

            samples = self.resize.batch_resize(samples)  # Clean resize
            samples = self._augment_dimensions(samples)  # Adjust input dimensions
            features[c:c+b] = self._model_fwd(samples)  # Model fwd pass
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

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

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_dataset(self, dataset, num_samples,
                                      batch_size=128, num_workers=0,
                                      shuffle=False):
        logging.info(f"Computing features for {num_samples:,} samples from data set")

        # Determine dimensionality of data set targets
        if isinstance(dataset.targets, list):  # List
            target_shape = (num_samples, )  # One dimension
        elif type(dataset.targets) in [np.ndarray, torch.Tensor]:  # Numpy array or Tensor
            _, *target_dims = dataset.targets.shape  # Possibly multiple dimensions
            target_shape = (num_samples, *target_dims)
        else:  # Any other target data type not implemented
            raise NotImplementedError(f"Data set targets of type '{type(dataset.targets)}' currently not supported")

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=shuffle)
        dataiterator = iter(dataloader)
        features = torch.zeros((num_samples, self.num_features),
                               dtype=self.dtype, device=self.device)
        targets = torch.zeros(target_shape, dtype=dataset.targets.dtype,
                              device=self.device)

        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            samples, labels = next(dataiterator)  # Load samples and labels
            samples = samples[:b].to(self.device)  # Limit and convert

            samples = self.resize.batch_resize(samples)  # Clean resize
            samples = self._augment_dimensions(samples)  # Adjust input dimensions
            features[c:c+b] = self._model_fwd(samples)  # Compute and append
            targets[c:c+b] = labels[:b]  # Collect target labels

            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        if len(targets) > 0:
            self._targets = targets
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features, targets

    def save(self, path="./", name="features"):
        dir = self._get_path(path)
        dir.mkdir(exist_ok=True)  # Create save directory
        save_path = dir/f"{name}.pt"

        data = {'features': self.features, 'targets': self.targets}
        torch.save(data, save_path)

        logging.info(f"Features and targets saved to '{save_path}'")

    def set_log_level(self, log_level):
        assert log_level in log_levels.keys(), f"Log level {log_level} not available"
        logging.basicConfig(format='%(message)s', level=log_levels[log_level])

    def _get_path(self, path):
        # Ensure clean absolute Path object
        return Path(path).expanduser().resolve()

    def __repr__(self):
        return f"CleanFeatures, {self.model.name} embedding, {self.num_features} features"
