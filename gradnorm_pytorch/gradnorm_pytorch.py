from functools import cache, partial

import torch
import torch.distributed as dist
from torch.autograd import grad
import torch.nn.functional as F
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

# tensor helpers

def l1norm(t, dim = -1):
    return F.normalize(t, p = 1, dim = dim)

# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    return dist.all_reduce(t, dist.ReduceOp.MEAN)

# main class

class GradNormLossWeighter(Module):
    @beartype
    def __init__(
        self,
        loss_weights: Union[
            List[float],
            Tensor
        ],
        *,
        learning_rate = 1e-4,
        restoring_force_alpha = 0.,
        grad_norm_parameters: Optional[Parameter] = None,
        accelerator: Optional[Accelerator] = None,
        frozen = False
    ):
        super().__init__()

        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)

        assert loss_weights.ndim == 1, 'loss weights must be 1 dimensional'

        num_losses = loss_weights.numel()

        self.accelerator = accelerator
        self.num_losses = num_losses
        self.frozen = frozen

        self.alpha = restoring_force_alpha

        self._grad_norm_parameters = [grad_norm_parameters] # hack

        # loss weights, either learned or static

        self.register_buffer('loss_weights', loss_weights)

        self.learning_rate = learning_rate

        # todo: figure out how best to smooth L0 over course of training, in case initialization of network is not great

        self.register_buffer('initial_losses', torch.zeros(num_losses))

        # for renormalizing loss weights at end

        self.register_buffer('loss_weights_sum', self.loss_weights.sum())

        # step, for maybe having schedules etc

        self.register_buffer('step', torch.tensor(0.))

    @property
    def grad_norm_parameters(self):
        return self._grad_norm_parameters[0]

    def backward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @beartype
    def forward(
        self,
        losses: Union[
            List[Tensor],
            Tensor
        ],
        shared_activations: Optional[Tensor] = None,     # in the paper, they used the grad norm of penultimate parameters from a backbone layer. but this could also be activations (say shared image being fed to multiple discriminators)
        freeze = False,                                  # can additionally freeze a learnable network on forward
        **backward_kwargs
    ):
        # backward functions dependent on whether using hf accelerate or not

        backward = self.accelerator.backward if exists(self.accelerator) else lambda l: l.backward()
        backward = partial(backward, **backward_kwargs)

        # validate that all the losses are a single scalar

        assert all([loss.numel() == 1 for loss in losses])

        # cast losses to tensor form

        if isinstance(losses, list):
            losses = torch.stack(losses)

        assert losses.ndim == 1, 'losses must be 1 dimensional'

        # handle base frozen case, so one can freeze the weights after a certain number of steps, or just to a/b test against learned gradnorm loss weights

        if self.frozen or freeze or not self.training:
            total_weighted_loss = (losses * self.loss_weights.detach()).sum()
            backward(total_weighted_loss)
            return total_weighted_loss

        # store initial loss

        if self.step.item() == 0:
            initial_losses = maybe_distributed_mean(losses)
            self.initial_losses.copy_(initial_losses)

        # determine which tensor to get grad norm from

        grad_norm_tensor = default(shared_activations, self.grad_norm_parameters)

        assert exists(grad_norm_tensor), 'you need to either set `grad_norm_parameters` on init or `shared_activations` on backwards'

        grad_norm_tensor.requires_grad_()

        # get grad norm with respect to each loss

        grad_norms = []
        loss_weights = self.loss_weights.clone()
        loss_weights = Parameter(loss_weights)

        for weight, loss in zip(loss_weights, losses):
            gradients, = grad(weight * loss, grad_norm_tensor, create_graph = True, retain_graph = True)

            grad_norm = gradients.norm(p = 2)
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # main algorithm for loss balancing

        grad_norm_average = maybe_distributed_mean(grad_norms.mean())

        loss_ratio = losses.detach() / self.initial_losses

        relative_training_rate = l1norm(loss_ratio)

        gradient_target = (grad_norm_average * (relative_training_rate ** self.alpha)).detach()

        grad_norm_loss = F.l1_loss(grad_norms, gradient_target)

        backward(grad_norm_loss)

        # manually take a single gradient step

        updated_loss_weights = loss_weights - loss_weights.grad * self.learning_rate

        renormalized_loss_weights = l1norm(updated_loss_weights) * self.loss_weights_sum

        self.loss_weights.copy_(renormalized_loss_weights)

        # increment step

        self.step.add_(1)
