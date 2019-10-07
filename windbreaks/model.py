from loguru import logger

import segmentation_models_pytorch as smp


def get_model(model_name, model_params):
    model_dict = {
        'fpn': smp.FPN,
        'unet': smp.Unet,
        'linknet': smp.Linknet,
        'pspnet': smp.PSPNet
    }

    logger.debug(f'Loading model {model_name} with params: {model_params}')
    model = model_dict.get(model_name)(**model_params)
    return model
