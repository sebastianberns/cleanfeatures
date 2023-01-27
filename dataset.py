from pathlib import Path
from typing import Callable, Tuple, Union, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CleanFeaturesDataset(Dataset[Tuple[Tensor, Optional[Tensor]]]):
    r"""Clean features dataset wrapper

    Each sample will be retrieved by indexing the features tensor along the
    first dimension.

    Args:
        path (str, Path): path to the saved features file
        map_location (str or device)
    """
    features: Tensor
    targets: Tensor
    num_features: int

    def __init__(self, path: Union[str, Path], map_location: Optional[Union[
                 torch.device, str, bytes, dict, Callable]] = None) -> None:
        path = Path(path).expanduser().resolve()

        data = torch.load(path, map_location=map_location)
        self.features = data['features']
        self.targets = data['targets'] if 'targets' in data.keys() else None
        self.data = self.features  # Alias
        self.num_features = self.features.size(1)

        if self.targets is not None:
            assert self.features.size(0) == len(self.targets), (f"Size mismatch "
                f"between features ({self.features.size(0):,} samples) and "
                f"targets ({len(self.targets):,} labels)")

    def __getitem__(self, index) -> Tuple[Tensor, Optional[Tensor]]:
        if self.targets is not None:
            return self.features[index], self.targets[index]
        return self.features[index], None

    def __len__(self) -> int:
        return self.features.size(0)
