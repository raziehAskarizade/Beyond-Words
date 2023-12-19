# Fardin Rastakhiz @ 2023

import torch.nn


def create_mse_loss(**kwargs):
    return torch.nn.MSELoss()


def create_bce_loss(**kwargs):
    return torch.nn.BCELoss()


def create_ce_loss(**kwargs):
    return torch.nn.CrossEntropyLoss()