import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostBatchNorm(nn.Module):
    """Ghost BatchNorm used in TabNet (simplified implementation)."""

    def __init__(self, input_dim: int, momentum: float = 0.01, virtual_batch_size: int = 128, eps: float = 1e-3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum, eps=eps)
        self.virtual_batch_size = virtual_batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.size(0) <= self.virtual_batch_size:
            return self.bn(x)
        chunks = torch.chunk(x, math.ceil(x.size(0) / self.virtual_batch_size), dim=0)
        res = [self.bn(c) for c in chunks]
        return torch.cat(res, dim=0)


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_glu: int = 2, virtual_batch_size: int = 128):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(n_glu):
            fc = nn.Linear(dim, 2 * output_dim)
            gbn = GhostBatchNorm(2 * output_dim, virtual_batch_size=virtual_batch_size)
            layers.append((fc, gbn))
            dim = output_dim
        self.layers = nn.ModuleList([nn.ModuleList(l) for l in layers])
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for fc, gbn in self.layers:
            x = fc(out)
            x = gbn(x)
            a, b = x.chunk(2, dim=-1)
            x = a * torch.sigmoid(b)
            out = (x + out[..., : self.output_dim]) * math.sqrt(0.5)
        return out


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = 128, relaxation_factor: float = 1.5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size=virtual_batch_size)
        self.relaxation_factor = relaxation_factor

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior
        return F.softmax(x, dim=-1)


class TabNetClassifier(nn.Module):
    """Minimal TabNet classifier for downstream binary classification."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.5,
        virtual_batch_size: int = 128,
    ):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.shared_feat_transform = FeatureTransformer(
            input_dim=input_dim, output_dim=n_d + n_a, n_glu=2, virtual_batch_size=virtual_batch_size
        )

        self.step_feat_transforms = nn.ModuleList(
            [FeatureTransformer(input_dim=n_d + n_a, output_dim=n_d + n_a, virtual_batch_size=virtual_batch_size) for _ in range(n_steps)]
        )
        self.attentive_transforms = nn.ModuleList(
            [AttentiveTransformer(input_dim=n_a, output_dim=input_dim, virtual_batch_size=virtual_batch_size) for _ in range(n_steps)]
        )

        self.bn_input = GhostBatchNorm(input_dim, virtual_batch_size=virtual_batch_size)
        self.fc_out = nn.Linear(n_d, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        x = self.bn_input(x)

        prior = torch.ones_like(x)
        M_loss: Optional[torch.Tensor] = None
        out_agg = 0

        x_feat = self.shared_feat_transform(x)
        d = x_feat[:, : self.n_d]
        a = x_feat[:, self.n_d :]

        for step in range(self.n_steps):
            mask = self.attentive_transforms[step](a, prior)
            prior = prior * (self.gamma - mask)

            if M_loss is None:
                M_loss = torch.mean(torch.sum(mask * torch.log(mask + 1e-15), dim=1))
            else:
                M_loss += torch.mean(torch.sum(mask * torch.log(mask + 1e-15), dim=1))

            x_masked = mask * x
            x_step = self.step_feat_transforms[step](x_masked)
            d = x_step[:, : self.n_d]
            a = x_step[:, self.n_d :]

            out_agg = out_agg + F.relu(d)

        logits = self.fc_out(out_agg)
        return logits


