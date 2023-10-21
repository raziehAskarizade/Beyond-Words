import torch
from torch import nn
import torch.nn.functional as functional


class TextToKG(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TextToKG, self).__init__(args, kwargs)


class SimpleTextToKG(TextToKG):
    def __init__(self, *args, **kwargs):
        super(SimpleTextToKG, self).__init__(args, kwargs)

