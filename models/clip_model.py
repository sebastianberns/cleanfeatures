import logging
from pathlib import Path

import clip
import torch
from torch import nn
from torchvision.transforms import Normalize


_MODELS = {
    # Resnet
    "RN50": {
        "url": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "filename": "RN50.pt",
        "input_size": 224,
        "embed_dims": 1024
    },
    "RN101": {
        "url": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "filename": "RN101.pt",
        "input_size": 224,
        "embed_dims": 512
    },
    "RN50x4": {
        "url": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "filename": "RN50x4.pt",
        "input_size": 288,
        "embed_dims": 640
    },
    "RN50x16": {
        "url": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "filename": "RN50x16.pt",
        "input_size": 384,
        "embed_dims": 768
    },
    "RN50x64": {
        "url": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "filename": "RN50x64.pt",
        "input_size": 448,
        "embed_dims": 1024
    },

    # Vision transformer
    "ViT-B/32": {
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "filename": "ViT-B-32.pt",
        "input_size": 224,
        "embed_dims": 512
    },
    "ViT-B/16": {
        "url": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "filename": "ViT-B-16.pt",
        "input_size": 224,
        "embed_dims": 512
    },
    "ViT-L/14": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "filename": "ViT-L-14.pt",
        "input_size": 224,
        "embed_dims": 768
    },
    "ViT-L/14@336px": {
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "filename": "ViT-L-14-336px.pt",
        "input_size": 336,
        "embed_dims": 768
    }
}


class CLIP(nn.Module):
    """
    CLIP image embedding

        path (str): locally saved model checkpoint
        device (str or device, optional): which device to load the model checkpoint onto
        clip_model (str, optional): name of clip model variant. Default: 'ViT-L/14'
    """
    def __init__(self, path='./models', device=None, clip_model='ViT-L/14'):
        super().__init__()

        path = Path(path)  # Make sure this is a Path object

        assert clip_model in clip.available_models(), f"CLIP model '{clip_model}' not available"

        self.name = f"CLIP {clip_model}"
        self.input_channels = 3
        self.dtype = torch.float32

        config = _MODELS[clip_model]
        self.url = config['url']
        filename = config['filename']
        self.input_width = config['input_size']
        self.input_height = config['input_size']
        self.num_features = config['embed_dims']

        self.device = device

        # Check if model is already downloaded
        clip_path = path/filename   # Build path to checkpoint file
        if clip_path.is_file():     # if it is a file
            logging.info(f"Found {self.name} checkpoint in {path}")
            clip_model = clip_path  # then load it
        else:  # Otherwise, download checkpoint
            logging.info(f"Downloading {self.name} to {path}")

        # download the network if it is not present at the given directory
        self.base, _ = clip.load(clip_model, device=device, jit=True,
                                 download_root=path)

        # CLIP preprocessing normalization
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
        self.normalization = Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))


    """
    Compute clip image embeddings

        input (tensor [B, C, W, H]): batch of image tensors

    Returns a tensor of feature embeddings [B, 768]
    """
    def forward(self, input):
        # Make sure input matches expected dimensions
        B, C, W, H = input.shape  # Batch size, channels, width, height
        assert (W == self.input_width) and (H == self.input_height)

        input = self.normalization(input)
        features = self.base.encode_image(input)
        return features
