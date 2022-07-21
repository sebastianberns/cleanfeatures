from pathlib import Path

import torch
from torch import nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models import Inception3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class InceptionV3(nn.Module):
    """
    Wrapper around Inception V3 torchvision model

        path (str): locally saved model checkpoint
        device (str or device, optional): which device to load the model checkpoint onto
        progress (bool, optional): display download progress bar. Default: True
        inception_layer (tuple[str, int], optional): tuple of name of layer for feature
            extraction and expected number of features. Default: ('avgpool', 2048)
    """
    def __init__(self, path='./models', device=None, progress=True, inception_layer=('avgpool', 2048)):
        super().__init__()

        path = Path(path)  # Make sure this is a Path object

        self.name = "InceptionV3"
        self.input_channels = 3
        self.input_width = 299
        self.input_height = 299

        self.layer, self.num_features = inception_layer

        self.device = device

        # Reproducing most of the default behavior
        # https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py#L432

        weights = Inception_V3_Weights.IMAGENET1K_V1  # Model checkpoint config
        weights.transforms = None  # No pre-processing image resize or crop

        # Initialize model instance
        # When loading weights ported from TF, inputs need to be transformed
        # https://github.com/pytorch/vision/issues/4136#issuecomment-871290495
        self.base = Inception3(transform_input=True, aux_logits=True,
            init_weights=False, num_classes=len(weights.meta["categories"])
        ).eval()

        # Load pre-trained model checkpoint
        checkpoint = load_state_dict_from_url(weights.url, model_dir=path,
            map_location=device, progress=progress)
        self.base.load_state_dict(checkpoint)

        # Create feature extractor
        return_nodes = {self.layer: 'features'}  # Define intermediate node output
        self.embedding = create_feature_extractor(self.base,
            return_nodes=return_nodes, suppress_diff_warning=True)


    """
    Compute inception features

        input (tensor [B, C, W, H]): batch of image tensors

    Returns a tensor of feature embeddings [B, 2048]
    """
    def forward(self, input):
        # Make sure input matches expected dimensions
        B, C, W, H = input.shape  # Batch size, channels, width, height
        assert (W == self.input_width) and (H == self.input_height)

        out = self.embedding(input)  # Forward pass, integrated normalization
        features = torch.flatten(out['features'], 1)  # Flatten, keep batch dim
        return features


    """
    Returns a list of layers that can be used for feature extraction
    """
    def get_layers(self):
        train_nodes, eval_nodes = get_graph_node_names(self.base,
            suppress_diff_warning=True)
        return eval_nodes
