from typing import Tuple, Optional
from math import log

import h5py
import torch
from torch.utils.data import Dataset


affinity_threshold = 1.0 - log(500) / log(50000)


class SequenceDataset(Dataset):

    def __init__(self, hdf5_path: str, classification: Optional[bool] = False):

        self._hdf5_path = hdf5_path

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            self._entry_names = list(hdf5_file.keys())

        self._classification = classification

    def __len__(self) -> int:
        return len(self._entry_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        entry_name = self._entry_names[index]

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:

            seq_embd = torch.tensor(hdf5_file[entry_name]["peptide/sequence_onehot"][:])

            if self._classification:

                target = torch.tensor((hdf5_file[entry_name]["affinity"][()] > (1.0 - log(500.0) / log(50000.0))).astype('int'), dtype=torch.long)
            else:
                target = torch.tensor([hdf5_file[entry_name]["affinity"][()]], dtype=torch.float)

        return seq_embd, target

