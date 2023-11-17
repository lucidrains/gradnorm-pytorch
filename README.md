<img src="./gradnorm.png" width="400px"></img>

## GradNorm - Pytorch (wip)

A practical implementation of <a href="https://arxiv.org/abs/1711.02257">GradNorm</a>, Gradient Normalization for Adaptive Loss Balancing, in Pytorch

Increasingly starting to come across neural network architectures that require more than 3 auxiliary losses, so will build out an installable package that easily handles loss balancing in distributed setting, gradient accumulation, etc. Also open to incorporating any follow up research; just let me know in the issues.

Will be dog-fooded for <a href="http://github.com/lucidrains/audiolm-pytorch">SoundStream</a>, <a href="https://github.com/lucidrains/magvit2-pytorch">MagViT2</a> as well as <a href="https://github.com/lucidrains/metnet-3">MetNet3</a>

## Install

```bash
$ pip install gradnorm-pytorch
```

## Usage

Basic static loss weighting

```python
import torch

from gradnorm_pytorch import (
    GradNorm,
    MockNetworkWithMultipleLosses
)

network = MockNetworkWithMultipleLosses(
    dim = 512,
    num_losses = 4
)

x = torch.randn(2, 512)

gradnorm = GradNorm(
    [1., 1., 1., 1.],
    frozen = True
)

total_loss, _ = network(x)

gradnorm.backward(total_loss)
```

## Citations

```bibtex
@article{Chen2017GradNormGN,
    title   = {GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks},
    author  = {Zhao Chen and Vijay Badrinarayanan and Chen-Yu Lee and Andrew Rabinovich},
    journal = {ArXiv},
    year    = {2017},
    volume  = {abs/1711.02257},
    url     = {https://api.semanticscholar.org/CorpusID:4703661}
}
```
