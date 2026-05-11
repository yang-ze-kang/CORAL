import os
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed: int = 666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_run_timestamp(fmt="%Y-%m-%d-%H-%M-%S") -> str:
    ts = os.environ.get("RUN_TIMESTAMP")
    if ts is None:
        ts = datetime.now().strftime(fmt)
        os.environ["RUN_TIMESTAMP"] = ts
    return ts