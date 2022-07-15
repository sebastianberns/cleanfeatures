from pathlib import Path
from typing import Callable, Union

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

    def __init__(self, path: Union[str, Path], device: Union[torch.device,
                 str, bytes, dict, Callable]) -> None:
        path = Path(path).expanduser().resolve()
        self.features = torch.load(path, map_location=device)

    def __getitem__(self, index) -> Tensor:
        return self.features[index]

    def __len__(self) -> int:
        return self.features.size(0)
