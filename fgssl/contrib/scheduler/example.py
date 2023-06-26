import math

from federatedscope.register import register_scheduler


def call_my_scheduler(optimizer, type):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if type == 'myscheduler':
        if optim is not None:
            lr_lambda = [lambda epoch: (epoch / 40) if epoch < 40 else 0.5 * (math.cos(40/100) * math.pi) + 1]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler


register_scheduler('myscheduler', call_my_scheduler)
