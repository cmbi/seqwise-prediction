from typing import Tuple
from math import log

import h5py
import torch
from torch.utils.data import Dataset


affinity_threshold = 1.0 - log(500) / log(50000)


class SequenceDataset(Dataset):

    def __init__(self, hdf5_path: str, blosum62: bool):

        self._hdf5_path = hdf5_path
        self.blosum62 = blosum62

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            self._entry_names = list(hdf5_file.keys())

    def __len__(self) -> int:
        return len(self._entry_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        entry_name = self._entry_names[index]

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:

            if self.blosum62:
                seq_embd = torch.zeros(9, 22)
                seq_embd[:, :20] = torch.tensor(hdf5_file[entry_name]["peptide/blosum62"][:])
            else:
                seq_embd = torch.tensor(hdf5_file[entry_name]["peptide/sequence_onehot"][:])
            ba = torch.tensor(hdf5_file[entry_name]["affinity"][()], dtype=torch.float)

        return seq_embd, ba

