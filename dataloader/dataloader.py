import os
import numpy as np
import fnmatch
import librosa as lib
import torch
from torch.utils.data import Dataset

from config import Config


from torch.utils.data import Dataset

import numpy as np
import os

op = Config()

def load_wav(path):
    wave_inputs, _ = lib.load(path, sr=op.sr)
    if len(wave_inputs) > op.dim:
        max_audio_start = len(wave_inputs) - op.dim
        audio_start = np.random.randint(0, max_audio_start)
        data = wave_inputs[audio_start: audio_start + op.dim]
    else:
        data = np.pad(wave_inputs, (0, op.dim - len(wave_inputs)), "constant")
    return data


class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths,loader=load_wav):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.loader = loader

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        return self.loader(noisy_file), self.loader(clean_file)

    def __len__(self):
        return len(self.noisy_paths)




