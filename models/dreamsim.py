from pathlib import Path

from dreamsim import dreamsim  # type: ignore[import]
from dreamsim.config import dreamsim_args  # type: ignore[import]
import torch
from torch import nn


class DreamSim(nn.Module):
    """
    DreamSim image embedding

        path (str): path to model checkpoint directory
        device (str or device, optional): which device to load the model checkpoint onto
    """

    def __init__(self, path='./models', device=None, dreamsim_type="ensemble"):
        super().__init__()

        path = Path(path) / 'dreamsim'  # Make sure this is a Path object
        self.device = device

        assert dreamsim_type in dreamsim_args['model_config'].keys(), f"DreamSim type '{dreamsim_type}' not available"

        self.name = f"DreamSim ({dreamsim_type})"
        self.input_channels = 3
        self.dtype = torch.float32
        img_size = dreamsim_args['img_size']  # 224
        self.input_width = img_size
        self.input_height = img_size

        self.model, _ = dreamsim(pretrained=True, device=device, cache_dir=path, 
                                 dreamsim_type=dreamsim_type)
        
        # Size of embedding depends on dreamsim_type
        # 'ensemble' -> 768 + 512 + 512 = 1792
        # 'dino_vitb16' -> 768
        # 'clip_vitb32' -> 512
        # 'open_clip_vitb32' -> 512
        self.num_features = self.model.embed_size


    """
    Compute dreamsim image embeddings

        input (tensor [B, C, W, H]): batch of image tensors

    Returns a tensor of feature embeddings [B, 768]
    """
    def forward(self, input):
        # Make sure input matches expected dimensions
        B, C, W, H = input.shape  # Batch size, channels, width, height
        assert (W == self.input_width) and (H == self.input_height)

        # Input is normalized internally in the model
        features = self.model.embed(input)
        return features
