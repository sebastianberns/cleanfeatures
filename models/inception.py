from pathlib import Path

import torch
from torch import nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models import Inception3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class InceptionV3(nn.Module):
    """
    Wrapper around Inception V3 torchvision model

        path (str): locally saved inception model snapshot
        device (str or device, optional): which device to load the model checkpoint onto
        progress (bool, optional): display download progress bar. Default: True
    """
    def __init__(self, path='./models', device=None, progress=True):
        super().__init__()

        path = Path(path)  # Make sure this is a Path object

        self.name = "InceptionV3"
        self.input_channels = 3
        self.input_width = 299
        self.input_height = 299
        self.num_features = 2048

        self.device = device

        # Reproducing most of the default behavior
        # https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py#L432

        weights = Inception_V3_Weights.IMAGENET1K_V1  # Model checkpoint config
        weights.transforms = None  # No image resize or crop

        # Initialize model instance
        self.base = Inception3(transform_input=False,  # No input normalization
            aux_logits=True, init_weights=False,
            num_classes=len(weights.meta["categories"])).eval()

        # Load pre-trained model checkpoint
        checkpoint = load_state_dict_from_url(weights.url, model_dir=path,
            map_location=device, progress=progress)
        self.base.load_state_dict(checkpoint)

        # Create feature extractor
        return_nodes = {'flatten': 'features'}  # Define intermediate node output
        self.embedding = create_feature_extractor(self.base,
            return_nodes=return_nodes, suppress_diff_warning=True)


    """
    Get the inception features without resizing

        x (tensor [B, C, W, H]): Image with values in range (0, 255)

    Returns a tensor of feature embeddings [B, 2048] of feature embeddings
    """
    def forward(self, input):
        B, C, W, H = input.shape  # Batch size, channels, width, height

        # Make sure input matches expected dimensions
        assert (W == self.input_width) and (H == self.input_height)

        # Normalization
        # Change value range from (0, 255) to (-1, +1)
        input = input - 128.  # Center around zero
        input = input / 128.  # Scale to (-1, +1)

        out = self.embedding(input)  # Forwar
        return out['features']
