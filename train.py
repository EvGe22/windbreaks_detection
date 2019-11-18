"""

"""

from collections import OrderedDict
from argparse import ArgumentParser
from typing import Dict, Any, Union

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim import Adam, SGD

from windbreaks.config import parse_config
from windbreaks.model import get_model
from windbreaks.data import WindbreakDataset

import safitty
from catalyst.dl import SupervisedWandbRunner

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import BCEDiceLoss, BCEJaccardLoss, DiceLoss, JaccardLoss


def parse_args():
    parser = ArgumentParser('', usage=__doc__)

    parser.add_argument('--config_path', default='default_config.yml',
                        required=False, help='Path to training config file')
    return parser.parse_args()


def get_loaders(train_config: Dict[str, Union[Dict, str]], valid_config) -> "OrderedDict[str, DataLoader]":
    train_dataset = WindbreakDataset(**train_config)
    valid_dataset = WindbreakDataset(**valid_config)

    return OrderedDict({
        "train": DataLoader(train_dataset, batch_size=1),
        "valid": DataLoader(valid_dataset, batch_size=1)
    })


def get_criterion(criterion_name, criterion_params):
    criterions = {
        'bce_dice': BCEDiceLoss,
        'bce_jaccard': BCEJaccardLoss,
        'dice': DiceLoss,
        'jaccard': JaccardLoss
    }
    if criterion_name not in criterions.keys():
        raise KeyError(f'{criterion_name} is not a valid criterion. Please provide one of: {criterions.keys()}')
    return criterions[criterion_name](**criterion_params)


def get_optimizer(optimizer_name, optimizer_params, model_):
    optimizers = {
        'adam': Adam,
        'sgd': SGD
    }
    default_lr = optimizer_params.get('lr', 1e-4)

    optimizer = optimizers[optimizer_name]([
        {'params': model_.decoder.parameters(), 'lr': optimizer_params.get('decoder_lr', default_lr)},
        {'params': model_.encoder.parameters(), 'lr': optimizer_params.get('encoder_lr', default_lr)},
    ])
    return optimizer


def get_scheduler(scheduler_name, scheduler_params, optimizer_) -> Union[ReduceLROnPlateau, _LRScheduler]:
    schedulers = {
        'reduce_on_plateau': ReduceLROnPlateau
    }
    if scheduler_name not in schedulers.keys():
        raise KeyError(f'{scheduler_name} is not a valid criterion. Please provide one of: {schedulers.keys()}')

    return schedulers[scheduler_name](**scheduler_params, optimizer=optimizer_)


# @TODO: add metrics support 
# (catalyst expects logits, rather than sigmoid outputs)
# metrics = [
#     smp.utils.metrics.IoUMetric(eps=1.),
#     smp.utils.metrics.FscoreMetric(eps=1.),
# ]

if __name__ == '__main__':
    args = parse_args()
    config = safitty.load(args.config_path)

    runner = SupervisedWandbRunner()

    model = get_model(
        model_name=safitty.get(config, 'model', 'name', default='unet'),
        model_params=safitty.get(config, 'model', 'params', default={}))

    criterion = get_criterion(
        criterion_name=safitty.get(config, 'criterion', 'name', default='bce_dice'),
        criterion_params=safitty.get(config, 'criterion', 'params', default={}))

    optimizer = get_optimizer(
        optimizer_name=safitty.get(config, 'optimizer', 'name', default='adam'),
        optimizer_params=safitty.get(config, 'optimizer', 'params', default={}),
        model_=model)

    scheduler = get_scheduler(
        scheduler_name=safitty.get(config, 'scheduler', 'name', default='reduce_on_plateau'),
        scheduler_params=safitty.get(config, 'scheduler', 'params', default={}),
        optimizer_=optimizer)

    loaders = get_loaders(
        train_config=safitty.get(config, 'data', 'train'),
        valid_config=safitty.get(config, 'data', 'valid'))

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=safitty.get(config, 'logdir', default='./log'),
        scheduler=scheduler,
        num_epochs=safitty.get(config, 'num_epochs', default=10),
        verbose=False,
        monitoring_params={
            'project': "geo_diplom"
        }
    )
