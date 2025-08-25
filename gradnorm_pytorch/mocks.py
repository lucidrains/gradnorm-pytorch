from torch import nn

class MockNetworkWithMultipleLosses(nn.Module):
    def __init__(
        self,
        dim,
        num_losses = 2
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.discriminators = nn.ModuleList([
            nn.Linear(dim, 1) for _ in range(num_losses)
        ])

    def forward(self, x, return_backbone_outputs = False):
        backbone_output = self.backbone(x)

        losses = []

        for discr in self.discriminators:
            loss = discr(backbone_output)
            losses.append(loss.mean())

        if not return_backbone_outputs:
            return losses

        return losses, backbone_output
