#!/usr/bin/env python

import logging
from pathlib import Path

import torch

from .inception_torchscript import InceptionV3W
from .resize import Resizer


"""
Standard Python logging levels
https://docs.python.org/3/library/logging.html#levels

Higher levels (lower is value) include all lower levels.
E.g. 'critical' only shows critical messages, whereas 'warning' will show
warnings, errors and critical messages.
"""
loglevels = {
    'all': 0,  # Log all messages
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 40,  # Log both errors and critical messages
    'critical': 50  # Only log critical messages
}


class CleanFeatures:
    """
    Initialize clean features processor

        path (str or Path): Path to snapshot of embedding model (i.e. Inception)
        device (str or device): Device to run model on (e.g. 'cuda' or 'cpu')
    """
    def __init__(self, model_path='./models', device=None, log='warning'):
        model_path = Path(model_path)  # Make sure this is a Path object

        # If device == None, set to cuda if available, otherwise set to cpu
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        assert log in loglevels.keys(), f"Log level {log} not available."
        logging.basicConfig(format='%(message)s', level=loglevels[log])

        logging.info('Loading model')
        self.model = InceptionV3W(model_path, device=self.device)
        self.num_features = self.model.num_features

        logging.info('Building resizer')
        self.resizer = Resizer(channels=self.model.channels,
                               width=self.model.width,
                               height=self.model.height)

        logging.info('CleanFeatures ready.')

    """
    Direct call to CleanFeatures instance

        ```python
        cf = CleanFeatures(path)
        feature = cf(input)
        ```
    """
    __call__ = _handle_input

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

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features(self, input):
        logging.info("Resizing ...")
        input = self.resizer(input)  # Clean resize

        logging.info("Adjusting number of dimensions ...")
        input = self.augment_dimensions(input)  # Adjust input dimensions

        logging.info("Model forward pass ...")
        features = self._model_fwd(input)  # Embed model forward pass
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
        num_features = self.num_features
        features = torch.zeros((num_samples, num_features),
                               device=self.device)
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - counter))  # Get batch size ...
                                # last batch may need to be smaller

            z = torch.randn((b, z_dim), device=self.device)  # Random samples
            with torch.no_grad():
                samples = generator(z)  # Generate images

            samples = self.resizer.batch_resize(samples)  # Clean resize
            features[c:c+b] = self._model_fwd(samples)  # Model fwd pass
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples
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
        num_features = self.num_features
        features = torch.zeros((num_samples, num_features),
                               device=self.device)
        c = 0  # Counter
        while c < num_samples:  # Until enough samples have been collected
            b = min(batch_size, (num_samples - counter))  # Get batch size ...
                                # last batch may need to be smaller

            samples, _ = next(dataiterator)  # Load samples
            samples = samples[:b].to(self.device)  # Limit and convert

            samples = self.resizer.batch_resize(samples)  # Clean resize
            features[c:c+b] = self._model_fwd(samples)  # Model fwd pass
            c += b  # Increase counter
        # Loop breaks when counter is equal to requested number of samples
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features
