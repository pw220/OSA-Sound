from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleItem:
    waveform: np.ndarray
    acoustic: np.ndarray
    demo: np.ndarray
    label: Optional[int] = None
    stage_label: Optional[int] = None


class ApneaSoundDataset(Dataset):
    """Generic dataset wrapper for apnea sound segments.

    - waveform: (T,)
    - acoustic: (D_ac,)
    - demo: (D_demo,)
    - label: scalar (0/1)
    """

    def __init__(self, samples: Dict[str, SampleItem]):
        self.keys = list(samples.keys())
        self.samples = samples

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key = self.keys[idx]
        item = self.samples[key]

        waveform = torch.from_numpy(item.waveform).float()
        acoustic = torch.from_numpy(item.acoustic).float()
        demo = torch.from_numpy(item.demo).float()
        label = -1 if item.label is None else int(item.label)
        stage_label = -1 if item.stage_label is None else int(item.stage_label)

        return {
            "waveform": waveform,
            "acoustic": acoustic,
            "demo": demo,
            "label": torch.tensor(label, dtype=torch.long),
            "stage_label": torch.tensor(stage_label, dtype=torch.long),
        }


