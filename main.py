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

        logging.info('Building resizer')
        self.resizer = Resizer()

        logging.info('Loading model')
        self.model = InceptionV3W(model_path, device=self.device)

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
        dims = len(input.shape)
        if dims == 4:
            logging.info("Assuming input data is batch of images")
            return compute_features_batch(input)
        elif dims == 3:
            logging.info("Assuming input data is single image")
            return compute_features_image(input)
        elif dims == 2:
            logging.info("Assuming input data individual channel")
            return compute_features_channel(input)
        else:
            raise ValueError(f"Input with {dims} dimensions is not supported")

    """
    Compute features of batch

        images (Pytorch tensor [B, C, W, H]): Batch of images with values in range (0, 255)

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_batch(self, batch):
        logging.info("Computing features for {0} images of {2} x {3} px in {1} channels".format(*batch.shape))

        logging.info("Resizing ...")
        batch_resized = self.resizer.batch_resize(batch)  # Clean resize of images to match expected model input
        logging.info("Resized images to {2} x {3} px.".format(*batch_resized.shape))

        logging.info("Model forward pass ...")
        features = self._model_fwd(batch_resized)  # Embed model forward pass
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    """
    Compute features of single image

        images (Pytorch tensor [C, W, H]): Single image with values in range (0, 255)

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_image(self, image):
        logging.info("Computing features for single image of {1} x {2} px in {0} channels".format(*image.shape))

        logging.info("Resizing ...")
        image_resized = self.resizer.image_resize(image)  # Clean resize of images to match expected model input
        logging.info("Resized image to {1} x {2} px.".format(*image_resized.shape))

        # Augment data to match model input dimensions
        image_resized = self.augment_dimensions(image_resized)

        logging.info("Model forward pass ...")
        features = self._model_fwd(image_resized)  # Embed model forward pass
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    """
    Compute features of individual channel

        images (Pytorch tensor [W, H]): Individual channel with values in range (0, 255)

    Returns a tensor of features [B, F] in range (-1, +1),
    where F is the number of features
    """
    def compute_features_channel(self, channel):
        logging.info("Computing features for individual channel of {0} x {1} px".format(*channel.shape))

        logging.info("Resizing ...")
        channel_resized = self.resizer.image_resize(channel)  # Clean resize of images to match expected model input
        logging.info("Resized channel to {0} x {1} px.".format(*channel_resized.shape))

        # Augment data to match model input dimensions
        image_resized = self.augment_dimensions(image_resized)

        logging.info("Model forward pass ...")
        features = self._model_fwd(channel_resized)  # Embed model forward pass
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
        num_features = self.model.num_features
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
        num_features = self.model.num_features
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
