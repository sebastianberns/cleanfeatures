from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CleanFeaturesDataset(Dataset[Tensor]):
    r"""Clean features dataset wrapper

    Each sample will be retrieved by indexing the features tensor along the
    first dimension.

    Args:
        path (str, Path): path to the saved features file
    """
    features: Tensor

    def __init__(self, path: Union[str, Path]) -> None:
        path = Path(path).expanduser().resolve()
        self.features = torch.load(path)

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.features.size(0)
