import torch
import torch.distributed as dist
from torch.autograd import grad
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, Parameter

from einops import rearrange

from accelerate import Accelerator

from beartype import beartype
from beartype.typing import Optional, Union, List

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class GradNormLossWeighting(Module):
    @beartype
    def __init__(
        self,
        loss_weights: Union[
            List[float],
            Tensor
        ],
        *,
        grad_norm_layer: Optional[Parameter] = None,
        accelerator: Optional[Accelerator] = None,
        frozen = False
    ):
        super().__init__()

        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)

        assert loss_weights.ndim == 1, 'loss weights must be 1 dimensional'
        assert frozen, 'only frozen implemented'

        num_losses = loss_weights.numel()

        self.accelerator = accelerator
        self.num_losses = num_losses
        self.frozen = frozen

        self._grad_norm_layer = [grad_norm_layer] # hack

        # loss weights, either learned or static

        self.loss_weights = Parameter(loss_weights, requires_grad = not frozen)

        # todo: figure out how best to smooth L0 over course of training, in case initialization of network is not great

        self.register_buffer('initial_loss', torch.zeros(num_losses))

        # step, for maybe having schedules etc

        self.register_buffer('step', torch.tensor(0.))

    @property
    def grad_norm_layer(self):
        return self._grad_norm_layer[0]

    def backward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @beartype
    def forward(
        self,
        losses: Union[
            List[Tensor],
            Tensor
        ],
        shared_tensor: Optional[Tensor] = None,     # in the paper, they used the grad norm of penultimate parameters from a backbone layer. but this could also be activations (say shared image being fed to multiple discriminators)
        freeze = False                              # can additionally freeze a learnable network on forward
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

        if self.frozen or freeze:
            total_weighted_loss = (losses * self.loss_weights.detach()).sum()
            backward(total_weighted_loss)
            return total_weighted_loss

        # determine which tensor to get grad norm from

        grad_norm_tensor = default(shared_tensor, self.grad_norm_layer)

        assert grad_norm_tensor.requires_grad, 'requires grad must be turned on for the tensor for deriving grad norm'

        # increment step

        self.step.add_(1)

        raise NotImplementedError
