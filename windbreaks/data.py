import os

from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import pandas as pd
from glob import glob


def to_tensor(x):
    return x.transpose(2, 0, 1).astype('float32')


class WindbreakDataset(Dataset):

    def __init__(self, data_files=None, mask_files=None, data_csv_path=None, augmentation=False):
        if data_csv_path is not None:
            df = pd.read_csv(data_csv_path)
            data_files = df['tile_paths']
            mask_files = df['mask_paths']
        if data_files is None or mask_files is None:
            raise ValueError('Must provide either data_files and mask_files or a data_csv_path')
        self.data_files = data_files
        self.mask_files = mask_files

    def __getitem__(self, i):
        im = cv2.imread(self.data_files[i])   # nrg -> grn
        mask = cv2.imread(self.mask_files[i], 0) / 255

        return to_tensor(im), np.expand_dims(mask.astype('float32'), axis=0)

    def __len__(self):
        return len(self.data_files)
