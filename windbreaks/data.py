import imageio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

import albumentations as albu
from albumentations.pytorch import ToTensorV2


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.CoarseDropout(p=0.3),
        albu.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.3
        ),
        albu.GridDistortion(p=0.3),
    ]
    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


aug_dict = {
    'train': compose([hard_transforms(), post_transforms()]),
    'eval': compose([post_transforms()])
}


def to_tensor(x):
    return x.transpose(2, 0, 1).astype('float32')


class WindbreakDataset(Dataset):

    def __init__(self, data_files=None, mask_files=None, data_csv_path=None, augmentations=None):
        if data_csv_path is not None:
            df = pd.read_csv(data_csv_path)
            data_files = df['tile_paths']
            mask_files = df['mask_paths']
        if data_files is None or mask_files is None:
            raise ValueError('Must provide either data_files and mask_files or a data_csv_path')
        self.data_files = data_files
        self.mask_files = mask_files
        self.augmentations = aug_dict.get(augmentations)

    def __getitem__(self, i):
        im = np.array(imageio.imread(self.data_files[i]))  # nrg -> grn
        mask = np.array(imageio.imread(self.mask_files[i])) / 255
        if self.augmentations is not None:
            result = self.augmentations(image=im, mask=mask)
            return result['image'], result['mask']
        return to_tensor(im), np.expand_dims(mask.astype('float32'), axis=0)

    def __len__(self):
        return len(self.data_files)
