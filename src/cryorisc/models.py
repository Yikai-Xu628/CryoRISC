from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from escnn import gspaces
from escnn import nn as enn


def regular_type(space, fields: int):
    return enn.FieldType(space, [space.regular_repr] * fields)


class CryoC8Encoder(nn.Module):


    def __init__(
        self, embedding_dim: int = 64, rotations: int = 8,
        full_c16: bool = False, field_multiplier: float = 1.0,
        residual_blocks: int = 0,
        multiscale_radial: bool = False, radial_bins: int = 4,
        preserve_regular_fields: bool = False,
    ):
        super().__init__()
        self.space = gspaces.rot2dOnR2(N=rotations)
        self.input_type = enn.FieldType(self.space, [self.space.trivial_repr])


        if rotations >= 16 and full_c16:
            scale = 1.0 if preserve_regular_fields else 16.0 / rotations
        else:
            scale = 8.0 / rotations
        type1 = regular_type(self.space, max(1, round(4 * scale * field_multiplier)))
        type2 = regular_type(self.space, max(1, round(8 * scale * field_multiplier)))
        type3 = regular_type(self.space, max(1, round(12 * scale * field_multiplier)))

        self.multiscale_radial = multiscale_radial
        self.radial_bins = radial_bins
        feature_modules = (
            enn.R2Conv(self.input_type, type1, kernel_size=7, padding=3, bias=False),
            enn.InnerBatchNorm(type1),
            enn.ReLU(type1, inplace=True),
            enn.PointwiseAvgPool(type1, kernel_size=2, stride=2),
            enn.R2Conv(type1, type2, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(type2),
            enn.ReLU(type2, inplace=True),
            enn.PointwiseAvgPool(type2, kernel_size=2, stride=2),
            enn.R2Conv(type2, type3, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type3),
            enn.ReLU(type3, inplace=True),
            enn.PointwiseAvgPool(type3, kernel_size=2, stride=2),
        )
        if multiscale_radial:


            self.stage1 = enn.SequentialModule(*feature_modules[:4])
            self.stage2 = enn.SequentialModule(*feature_modules[4:8])
            self.stage3 = enn.SequentialModule(*feature_modules[8:])
            self.stage_group_pools = nn.ModuleList([
                enn.GroupPooling(type1), enn.GroupPooling(type2),
                enn.GroupPooling(type3),
            ])
        else:
            self.features = enn.SequentialModule(*feature_modules)
            self.group_pool = enn.GroupPooling(type3)
        self.residual_blocks = nn.ModuleList([
            enn.SequentialModule(
                enn.R2Conv(type3, type3, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(type3),
                enn.ReLU(type3, inplace=True),
                enn.R2Conv(type3, type3, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(type3),
            )
            for _ in range(residual_blocks)
        ])
        self.residual_relu = enn.ReLU(type3, inplace=True)
        if multiscale_radial:
            projector_input = sum(
                pool.out_type.size for pool in self.stage_group_pools
            ) * radial_bins
        else:
            projector_input = self.group_pool.out_type.size
        self.projector = nn.Sequential(
            nn.Linear(projector_input, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, embedding_dim),
        )

    @staticmethod
    def radial_pool(features: torch.Tensor, bins: int) -> torch.Tensor:

        _, _, height, width = features.shape
        yy = torch.linspace(-1.0, 1.0, height, device=features.device)
        xx = torch.linspace(-1.0, 1.0, width, device=features.device)
        radius = torch.sqrt(yy[:, None].square() + xx[None, :].square())

        descriptors = []
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            mask = (radius >= lower) & (radius < upper if index + 1 < bins else radius <= upper)
            weight = mask.to(features.dtype)
            descriptors.append(
                (features * weight).sum(dim=(-2, -1)) / weight.sum().clamp_min(1.0)
            )
        return torch.cat(descriptors, dim=1)

    def forward(self, images):
        x = enn.GeometricTensor(images, self.input_type)
        if self.multiscale_radial:
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x = self.stage3(x2)
        else:
            x = self.features(x)
        for block in self.residual_blocks:
            residual = block(x)
            x = self.residual_relu(enn.GeometricTensor(x.tensor + residual.tensor, x.type))
        if self.multiscale_radial:
            x = torch.cat([
                self.radial_pool(pool(level).tensor, self.radial_bins)
                for pool, level in zip(self.stage_group_pools, (x1, x2, x))
            ], dim=1)
        else:
            x = self.group_pool(x).tensor
            x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return F.normalize(self.projector(x), dim=1)


class ResidualMetricAdapter(nn.Module):
    def __init__(self, dimension: int, bottleneck: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, bottleneck), nn.LayerNorm(bottleneck), nn.SiLU(),
            nn.Linear(bottleneck, dimension),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, embedding):
        correction = self.net(embedding)
        return F.normalize(embedding + torch.sigmoid(self.gate) * correction, dim=1)


CryoRISCEncoder = CryoC8Encoder
