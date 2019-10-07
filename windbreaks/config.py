import json
from copy import deepcopy
from datetime import datetime


config_defaults = {
    'num_epochs': 10,
    'logdir': f"./logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}",

}


def parse_config(config_path):
    config = deepcopy(config_defaults)
    with open(config_path) as f:
        config.update(json.load(f))
    return config
