import os

from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from glob import glob


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


class WindbreakDataset(Dataset):

    def __init__(self, data_path, augmentation=False):
        self.data_path = data_path
        self.data_files = glob(os.path.join(self.data_path, '*_data.png'))
        self.mask_files = [f.replace('data.png', 'mask.png') for f in self.data_files]

    def __getitem__(self, i):
        im = cv2.imread(self.data_files[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[i], 0) / 255

        return to_tensor(im), np.expand_dims(mask.astype('float32'), axis=0)

    def __len__(self):
        return len(self.data_files)