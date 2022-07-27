from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CleanFeaturesDataset(Dataset[Tuple[Tensor, int]]):
    r"""Clean features dataset wrapper

    Each sample will be retrieved by indexing the features tensor along the
    first dimension.

    Args:
        path (str, Path): path to the saved features file
        map_location (str or device)
    """
    features: Tensor
    targets: List[int]

    def __init__(self, path: Union[str, Path], map_location: Union[
                 torch.device, str, bytes, dict, Callable]=None) -> None:
        path = Path(path).expanduser().resolve()

        data = torch.load(path, map_location=map_location)
        self.features = data['features']
        self.targets = data['targets']

        assert self.features.size(0) == len(self.targets), (f"Size mismatch "
            f"between features ({self.features.size(0)}) and "
            f"targets ({len(self.targets)})")

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        return self.features[index], self.targets[index]

    def __len__(self) -> int:
        return self.features.size(0)
