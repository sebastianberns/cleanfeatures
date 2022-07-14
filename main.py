#!/usr/bin/env python

import logging
from pathlib import Path

import torch

from . import models
from .resize import Resizer


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
    augment_dimensions: Standardize input shape to [B, 3, W, H]
    self: Compute features from any type of input
    compute_features: Direct access to processing pipeline
    compute_features_from_generator: Sample generator model
    compute_features_from_dataset: Get sample from dataloader
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
    def __init__(self, model_path='./models', model='InceptionV3W',
                 device=None, log='warning', **kwargs):

        # Check if model is implemented
        assert hasattr(models, model), f"Model {model} is not available"

        model_path = self._get_path(model_path)

        # If device == None, set to cuda if available, otherwise set to cpu
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.set_log_level(log)

        logging.info('Loading model')
        model_fn = getattr(models, model)  # Get executable from string
        self.model = model_fn(model_path, self.device, **kwargs)
        self.num_features = self.model.num_features

        logging.info('Building resizer')
        self.resizer = Resizer(channels=self.model.input_channels,
                               width=self.model.input_width,
                               height=self.model.input_height)

        self._features = None

        logging.info('CleanFeatures ready.')

    @property
    def features(self):
        return self._features

    """
    Redirect calls based on input data type

        input   Tensor: directly process input
                Module: sample batch from generator model
                DataLoader: load batch from data set
    """
    def _handle_input(self, input, *kwargs):
        if isinstance(input, torch.Tensor):  # Tensor ready for processing
            return self.compute_features(input)
        elif isinstance(input, torch.nn.Module):  # Generator model
            return self.compute_features_from_generator(input, *kwargs)
        elif isinstance(input, torch.utils.data.DataLoader):  # Data set
            return self.compute_features_from_dataset(input, *kwargs)
        else:
            raise ValueError(f"Input with {dims} dimensions is not supported")

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
    def augment_dimensions(self, input):
        channels = self.model.input_channels
        dims = len(input.shape)

        # Adjust number of dimensions
        if dims == 2:  # [W, H]
            input.unsqueeze_(0)  # Add channel dimension
            logging.info("Added channel dimension")
        if dims == 3:  # [C, W, H]
            input.unsqueeze_(0)  # Add batch dimension
            logging.info("Added batch dimension")
        # Now all input is standardized to [B, C, W, H]
        B, C, W, H = input.shape

        if C < channels:  # Grayscale image
            # Increase channel dimension with same data (just view, no copy)
            input = input.expand(-1, channels, -1, -1)
            logging.info(f"Expanded channel dimensions to {channels}")

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
        input = self.resizer(input)  # Clean resize

        logging.info("Adjusting number of dimensions ...")
        input = self.augment_dimensions(input)  # Adjust input dimensions

        logging.info("Model forward pass ...")
        features = self._model_fwd(input)  # Embed model forward pass

        self._features = features
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    """
    Compute features of samples from pre-trained generator

        generator (Module): Pre-trained generator model
        z_dim (int): Number of generator input dimensions
        num_samples (int): Number of samples to generate and process
        batch_size (int): Batch size for generator sampling

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_generator(self, generator, z_dim=512,
                                        num_samples=50_000, batch_size=128):
        logging.info(f"Computing features for {num_samples} samples from generator")
        generator.eval()
        features = torch.zeros((num_samples, self.num_features),
                               device=self.device)
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            z = torch.randn((b, z_dim), device=self.device)  # Random samples
            with torch.no_grad():
                samples = generator(z)  # Generate images

            samples = self.resizer.batch_resize(samples)  # Clean resize
            features[c:c+b] = self._model_fwd(samples)  # Model fwd pass
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    """
    Compute features of samples from data set

        dataloader (DataLoader): Instance of Pytorch data loader
        num_samples (int): Number of samples to process
        batch_size (int): Batch size for sampling

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_from_dataset(self, dataloader, num_samples=50_000,
                                      batch_size=128):
        logging.info(f"Computing features for {num_samples} samples from data set")
        dataiterator = iter(dataloader)
        features = torch.zeros((num_samples, self.num_features),
                               device=self.device)
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - c))  # Get batch size ...
                                # last batch may need to be smaller

            samples, _ = next(dataiterator)  # Load samples
            samples = samples[:b].to(self.device)  # Limit and convert

            samples = self.resizer.batch_resize(samples)  # Clean resize
            features[c:c+b] = self._model_fwd(samples)  # Model fwd pass
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples

        self._features = features
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    def save(self, path="./", name="features"):
        dir = self._get_path(path)
        dir.mkdir(exist_ok=True)  # Create save directory
        save_path = dir/f"{name}.pt"

        torch.save(self.features, save_path)
        logging.info(f"Features saved to '{save_path}'")

    def set_log_level(self, log_level):
        assert log_level in log_levels.keys(), f"Log level {log_level} not available"
        logging.basicConfig(format='%(message)s', level=log_levels[log_level])

    def _get_path(self, path):
        # Ensure clean absolute Path object
        return Path(path).expanduser().resolve()

    def __repr__(self):
        return f"CleanFeatures, {self.model.name} embedding, {self.num_features} features"
