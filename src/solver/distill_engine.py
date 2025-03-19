import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

def train_one_epoch(model: torch.nn.module,
                    teacher: torch.nn.module, criterion: torch.nn.module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    
    model.train()
    criterion.train()

    if teacher is not None:
        teacher.eval()
    else:
        raise NotImplementedError('Teacher model is required for distillation.')
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    