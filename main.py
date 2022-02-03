#!/usr/bin/env python

import logging

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
    def __init__(self, model_path, device=None, log='warning'):
        # If device == None, set to cuda if available, otherwise set to cpu
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        logging.basicConfig(format='%(message)s', level=loglevels[log])

        logging.info('Building resizer...')
        self.resizer = Resizer()

        logging.info('Loading model...')
        self.model = InceptionV3W(model_path, device=self.device)

        logging.info('CleanFeatures ready.')

    """
    Compute features of images in the embedding space of a deep computer
    vision model

        images (Pytorch tensor [B, C, W, H]): Batch of images with values in range (0, 255)

    Returns a tensor of the resized image batch [B, C, X, Y] in range (-1, +1),
    where X and Y are the resized width and height
    """
    def clean_features(self, images):
        logging.info("Computing features for {0} images of {2} x {3} px in {1} channels".format(*images.shape))

        logging.info('Resizing...')
        images_resized = self.resizer(images)  # Clean resize of images to match expected model input
        logging.info("Resized images to {2} x {3} px.".format(*images_resized.shape))

        logging.info('Model forward pass...')
        features = self.model(images_resized)  # Embed model forward pass
        logging.info("Computed features for {0} batch items in {1} dimensions.".format(*features.shape))
        return features

    """
    Direct call to CleanFeatures instance

        ```python
        cf = CleanFeatures(path)
        feature = cf(images)
        ```
    """
    __call__ = clean_features
