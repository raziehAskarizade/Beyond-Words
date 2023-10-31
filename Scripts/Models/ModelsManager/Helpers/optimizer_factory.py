import torch


def create_adam_optimizer(lr, model, **kwargs):
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
    return torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


def create_sgd_optimizer(lr, model, **kwargs):
    return torch.optim.SGD(model.parameters(), lr)


def create_rms_prop_optimizer(lr, model, **kwargs):
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
    return torch.optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)