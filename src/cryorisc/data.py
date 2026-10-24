from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ProjectionDataset(Dataset):


    def __init__(
        self, path: Path, noise_snr_min=None, noise_snr_max=None,
        blur_sigma=0.0, deterministic_noise=False, noise_seed=20260803,
        max_shift=0, scale_min=1.0, scale_max=1.0,
        return_structure=False,
    ):
        with np.load(path) as data:
            images = np.asarray(data["images"], dtype=np.float32)
            viewdirs = np.asarray(data["viewdirs"], dtype=np.float32)
            group_ids = np.asarray(data["group_ids"], dtype=np.int64) \
                if "group_ids" in data else np.arange(len(images), dtype=np.int64)
            structure_ids = np.asarray(data["structure_ids"], dtype=np.int64) \
                if "structure_ids" in data else np.zeros(len(images), dtype=np.int64)

        if images.ndim != 3:
            raise ValueError(f"Expected N x H x W projections, got {images.shape}")
        if viewdirs.shape != (len(images), 3):
            raise ValueError(
                f"Expected N x 3 viewing directions, got {viewdirs.shape}"
            )


        means = images.mean(axis=(1, 2), keepdims=True)
        stds = images.std(axis=(1, 2), keepdims=True)
        self.images = (images - means) / np.maximum(stds, 1e-6)
        self.viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=1, keepdims=True)
        self.group_ids = group_ids.reshape(-1)
        self.structure_ids = structure_ids.reshape(-1)
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.deterministic_noise = deterministic_noise
        self.noise_seed = noise_seed
        self.blur_sigma = blur_sigma
        self.max_shift = int(max_shift)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.return_structure = bool(return_structure)
        self.blur_kernel = None
        if blur_sigma > 0:
            radius = int(np.ceil(3 * blur_sigma))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel1d = torch.exp(-0.5 * (coords / blur_sigma).square())
            kernel1d /= kernel1d.sum()
            self.blur_kernel = torch.outer(kernel1d, kernel1d)[None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index]).unsqueeze(0)
        if abs(self.scale_max - self.scale_min) > 1e-6:
            if self.deterministic_noise:
                scale_rng = np.random.default_rng(self.noise_seed + index + 170003)
                scale = float(scale_rng.uniform(self.scale_min, self.scale_max))
            else:
                scale = float(np.random.uniform(self.scale_min, self.scale_max))
            image = scale_on_canvas(image, scale)
        if self.max_shift > 0:
            if self.deterministic_noise:
                shift_rng = np.random.default_rng(self.noise_seed + index + 99173)
                dx, dy = shift_rng.integers(-self.max_shift, self.max_shift + 1, size=2)
            else:
                dx, dy = np.random.randint(-self.max_shift, self.max_shift + 1, size=2)
            image = torch.roll(image, shifts=(int(dx), int(dy)), dims=(-2, -1))
        if self.noise_snr_min is not None:
            if self.deterministic_noise:
                rng = np.random.default_rng(self.noise_seed + index)
                uniform = rng.uniform()
                noise = torch.from_numpy(
                    rng.standard_normal(image.shape, dtype=np.float32)
                )
            else:
                uniform = np.random.uniform()
                noise = torch.randn_like(image)
            log_min = np.log(self.noise_snr_min)
            log_max = np.log(self.noise_snr_max)
            snr = float(np.exp(log_min + uniform * (log_max - log_min)))
            image = image + noise / (snr ** 0.5)
        if self.blur_kernel is not None:
            radius = self.blur_kernel.shape[-1] // 2
            padded = F.pad(image[None], (radius,) * 4, mode="reflect")
            image = F.conv2d(padded, self.blur_kernel)[0]
        if self.noise_snr_min is not None or self.blur_kernel is not None:
            image = (image - image.mean()) / image.std().clamp_min(1e-6)
        viewdir = torch.from_numpy(self.viewdirs[index])
        result = (
            image, viewdir, torch.tensor(self.group_ids[index], dtype=torch.long)
        )
        if self.return_structure:
            result += (torch.tensor(self.structure_ids[index], dtype=torch.long),)
        return result


def scale_on_canvas(image: torch.Tensor, scale: float) -> torch.Tensor:

    side = image.shape[-1]
    resized_side = max(1, round(side * scale))
    resized = F.interpolate(
        image[None], size=(resized_side, resized_side), mode="bilinear",
        align_corners=False, antialias=True,
    )[0]
    if resized_side >= side:
        start = (resized_side - side) // 2
        return resized[:, start:start+side, start:start+side]
    output = torch.zeros_like(image)
    start = (side - resized_side) // 2
    output[:, start:start+resized_side, start:start+resized_side] = resized
    return output

