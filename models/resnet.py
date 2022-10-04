from pathlib import Path

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.transforms import Normalize


# https://github.com/pytorch/vision/blob/v0.13.0/torchvision/models/resnet.py#L354
model_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


class Resnet50(nn.Module):
    """
    Wrapper around Resnet50 torchvision model

        path (str): locally saved model checkpoint
        device (str or device, optional): which device to load the model checkpoint onto
        progress (bool, optional): display download progress bar. Default: True
        resnet_layer (tuple[str, int], optional): tuple of name of layer for feature
            extraction and expected number of features. Default: ('avgpool', 2048)
    """
    def __init__(self, path='./models', device=None, progress=True, resnet_layer=('avgpool', 2048)):
        super().__init__()

        path = Path(path).expanduser().resolve()

        self.name = "Resnet50"
        self.input_channels = 3
        self.input_width = 224
        self.input_height = 224
        self.dtype = torch.float32

        self.layer, self.num_features = resnet_layer

        self.device = device

        # Reproducing most of the default behavior
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L699

        # Initialize model instance
        self.base = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        self.base.to(device)

        # Load pre-trained model checkpoint
        checkpoint = load_state_dict_from_url(model_url, model_dir=path,
            map_location=device, progress=progress)
        self.base.load_state_dict(checkpoint)
        self.base.eval()

        # Create feature extractor
        return_nodes = {self.layer: 'features'}  # Define intermediate node output
        self.embedding = create_feature_extractor(self.base,
            return_nodes=return_nodes, suppress_diff_warning=True)

        # Imagenet statistics
        self.normalization = Normalize((0.485, 0.456, 0.406),  # mean
                                       (0.229, 0.224, 0.225))  # std


    """
    Compute resnet features

        input (tensor [B, C, W, H]): batch of image tensors

    Returns a tensor of feature embeddings [B, 2048]
    """
    def forward(self, input):
        # Make sure input matches expected dimensions
        B, C, W, H = input.shape  # Batch size, channels, width, height
        assert (W == self.input_width) and (H == self.input_height)

        input = self.normalization(input)
        out = self.embedding(input)  # Forward pass
        features = torch.flatten(out['features'], 1)  # Flatten, keep batch dim
        return features


    """
    Returns a list of layers that can be used for feature extraction
    """
    def get_layers(self):
        train_nodes, eval_nodes = get_graph_node_names(self.base,
            suppress_diff_warning=True)
        return eval_nodes
