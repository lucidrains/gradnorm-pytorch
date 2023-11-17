import torch
import torch.distributed as dist
from torch.autograd import grad
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

    def backward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @beartype
    def forward(
        self,
        losses: Union[
            List[Tensor],
            Tensor
        ]
    ):
        # backward functions dependent on whether using hf accelerate or not

        backward = self.accelerator.backward if exists(self.accelerator) else lambda l: l.backward()

        # validate that all the losses are a single scalar

        assert all([loss.numel() == 1 for loss in losses])

        # cast losses to tensor form

        if isinstance(losses, list):
            losses = torch.stack(losses)

        assert losses.ndim == 1, 'losses must be 1 dimensional'

        # handle base frozen case, so one can freeze the weights after a certain number of steps, or just to a/b test against learned gradnorm loss weights

        if self.frozen:
            total_weighted_loss = (losses * self.loss_weights).sum()
            backward(total_weighted_loss)
            return total_weighted_loss

        raise NotImplementedError
