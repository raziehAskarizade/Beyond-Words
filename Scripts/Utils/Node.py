import torch
from typing import List, Tuple


class Node:

    def __init__(self,
                 value: torch.Tensor,
                 neighbors: List[torch.Tensor] | Tuple[torch.Tensor]):
        self.value = value
        self.neighbors = neighbors
