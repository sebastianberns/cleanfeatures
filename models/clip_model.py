import logging
from pathlib import Path

import torch
from torch.nn import Module
from torchvision.transforms import Normalize

from . import clip


_MODELS = {
    # Resnet
    "RN50": {
        "filename": "RN50.pt",
        "input_size": 224,
        "embed_dims": 1024
    },
    "RN101": {
        "filename": "RN101.pt",
        "input_size": 224,
        "embed_dims": 512
    },
    "RN50x4": {
        "filename": "RN50x4.pt",
        "input_size": 288,
        "embed_dims": 640
    },
    "RN50x16": {
        "filename": "RN50x16.pt",
        "input_size": 384,
        "embed_dims": 768
    },
    "RN50x64": {
        "filename": "RN50x64.pt",
        "input_size": 448,
        "embed_dims": 1024
    },

    # Vision transformer
    "ViT-B/32": {
        "filename": "ViT-B-32.pt",
        "input_size": 224,
        "embed_dims": 512
    },
    "ViT-B/16": {
        "filename": "ViT-B-16.pt",
        "input_size": 224,
        "embed_dims": 512
    },
    "ViT-L/14": {
        "filename": "ViT-L-14.pt",
        "input_size": 224,
        "embed_dims": 768
    },
    "ViT-L/14@336px": {
        "filename": "ViT-L-14-336px.pt",
        "input_size": 336,
        "embed_dims": 768
    }
}


class CLIP(Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

        path (str): locally saved inception model snapshot
    """
    def __init__(self, path='./models', device=None, clip_model='ViT-L/14'):
        super(CLIP, self).__init__()

        path = Path(path)  # Make sure this is a Path object

        assert clip_model in clip.available_models(), f"CLIP model '{clip_model}' not available"

        self.model_name = f"CLIP {clip_model}"
        self.input_channels = 3

        config = _MODELS[clip_model]
        filename = config['filename']
        self.input_width = config['input_size']
        self.input_height = config['input_size']
        self.num_features = config['embed_dims']

        self.device = device

        # Check if model is already downloaded
        clip_path = path/filename   # Build path to checkpoint file
        if clip_path.is_file():     # if it is a file
            logging.info(f"Found {self.model_name} checkpoint in {path}")
            clip_model = clip_path  # then load it
        else:  # Otherwise, download checkpoint
            logging.info(f"Downloading {self.model_name} to {path}")

        # download the network if it is not present at the given directory
        self.model, preprocess = clip.load(clip_model, device=device, jit=True,
                                           download_root=path)

        # CLIP preprocessing normalization
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
        self.normalization = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


    """
    Get the inception features without resizing

        x (Pytorch tensor [B, C, W, H]): Image with values in range (0, 255)

    Returns a Pytorch tensor [B, F] in range (-1, +1),
    where F is the number of features
    """
    def forward(self, x):
        input = self.normalization(x)
        features = self.model.encode_image(input)
        return features
