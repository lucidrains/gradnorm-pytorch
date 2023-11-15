<img src="./gradnorm.png" width="400px"></img>

## GradNorm - Pytorch (wip)

A practical implementation of <a href="https://arxiv.org/abs/1711.02257">GradNorm</a>, Gradient Normalization for Adaptive Loss Balancing, in Pytorch

Increasingly starting to come across neural network architectures that require more than 3 auxiliary losses, so will build out an installable package that easily handles loss balancing in distributed setting, gradient accumulation, etc. Also open to incorporating any follow up research; just let me know in the issues.

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
