from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(np.ceil(3 * sigma))
    x = torch.arange(
        -radius, radius + 1, dtype=images.dtype, device=images.device
    )
    kernel = torch.exp(-0.5 * (x / sigma).square())
    kernel /= kernel.sum()
    kernel = torch.outer(kernel, kernel)[None, None]
    padded = F.pad(images, (radius,) * 4, mode="reflect")
    return F.conv2d(padded, kernel)


def rescale_on_canvas(images: torch.Tensor, scale: float) -> torch.Tensor:

    if abs(scale - 1.0) < 1e-6:
        return images
    theta = torch.zeros((len(images), 2, 3), dtype=images.dtype, device=images.device)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def background_normalise(images: torch.Tensor, radius: float) -> torch.Tensor:
    side = images.shape[-1]
    coord = torch.arange(side, device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    centre = (side - 1.0) / 2.0
    background = torch.sqrt((yy - centre) ** 2 + (xx - centre) ** 2) >= radius
    mean = images[:, 0, background].mean(1, keepdim=True)
    std = images[:, 0, background].std(1, keepdim=True).clamp_min(1e-6)
    return (images - mean[:, None, None]) / std[:, None, None]


def spherical_kmeans(
    embeddings: np.ndarray,
    clusters: int,
    device: torch.device,
    seed: int,
    iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    count = len(embeddings)
    selected = np.empty(clusters, dtype=np.int64)
    selected[0] = int(rng.integers(count))
    closest_distance = np.maximum(
        0.0, 2.0 - 2.0 * (embeddings @ embeddings[selected[0]])
    )
    for index in range(1, clusters):
        total = float(closest_distance.sum())
        if total <= 1e-12:
            remaining = np.setdiff1d(np.arange(count), selected[:index], assume_unique=False)
            selected[index] = int(rng.choice(remaining))
        else:
            selected[index] = int(rng.choice(count, p=closest_distance / total))
        distance = np.maximum(
            0.0, 2.0 - 2.0 * (embeddings @ embeddings[selected[index]])
        )
        closest_distance = np.minimum(closest_distance, distance)

    values = torch.from_numpy(embeddings).to(device)
    centres = values[torch.from_numpy(selected).to(device)].clone()
    previous = None
    for iteration in range(iterations):
        similarity = values @ centres.T
        labels = similarity.argmax(1)
        occupancy = torch.bincount(labels, minlength=clusters)
        if bool((occupancy == 0).any()):
            confidence = similarity.max(1).values
            replacements = confidence.argsort()
            empty = torch.where(occupancy == 0)[0]
            labels[replacements[: len(empty)]] = empty
            occupancy = torch.bincount(labels, minlength=clusters)
        sums = torch.zeros_like(centres)
        sums.index_add_(0, labels, values)
        updated = F.normalize(sums, dim=1)
        change = float((1.0 - (centres * updated).sum(1)).abs().max())
        centres = updated
        if previous is not None and torch.equal(labels, previous):
            break
        previous = labels.clone()
        if iteration % 10 == 0 or change < 1e-6:
            print(f"spherical k-means iteration={iteration + 1} change={change:.6g}", flush=True)
        if change < 1e-6:
            break
    return labels.cpu().numpy().astype(np.int32), centres.cpu().numpy().astype(np.float32)

