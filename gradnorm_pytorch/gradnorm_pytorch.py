import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange

from accelerate import Accelerator

from beartype import beartype
from beartype.typing import Optional, Union, List

# helper functions

def exists(v):
    return v is not None

# main class

class GradNorm(Module):
    @beartype
    def __init__(
        self,
        loss_weights: Union[
            List[float],
            Tensor
        ],
        *,
        accelerator: Optional[Accelerator] = None,
        frozen = False
    ):
        super().__init__()

        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)

        assert loss_weights.ndim == 1, 'loss weights must be 1 dimensional'
        assert frozen, 'only frozen implemented'

        self.accelerator = accelerator
        self.num_losses = loss_weights.numel()
        self.frozen = frozen

        self.loss_weights = nn.Parameter(loss_weights, requires_grad = not frozen)

        self.register_buffer('step', torch.tensor(0.))

    @beartype
    def forward(
        self,
        losses: Union[
            List[Tensor],
            Tensor
        ]
    ):
        backwards = self.accelerator.backward if exists(self.accelerator.backward) else lambda l: l.backward()

        if isinstance(losses, list):
            losses = torch.stack(losses)

        assert losses.ndim == 1, 'losses must be 1 dimensional'

        total_weighted_loss = (losses * self.loss_weights).sum()

        # backward functions dependent on whether using hf accelerate or not

        backwards(total_weighted_loss)

        return total_weighted_loss
