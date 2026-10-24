from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import time
from .engine import circular_soft_mask, normalise_for_match, band_limited_for_match, read_mrc_memmap, rotate_batch, restricted_best_shift


@torch.no_grad()
def pixel_scores(templates, test, counts, out, emit, stack_path, batch_size=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = 60
    mask = circular_soft_mask(size, 65 / 3, 10 / 3, device)
    refs = F.interpolate(torch.from_numpy(templates).to(device)[:, None], size=(size, size),
                         mode='bilinear', align_corners=False, antialias=True)[:, 0]
    refs = normalise_for_match(band_limited_for_match(refs * mask, .30, .045), mask)
    fr = torch.fft.rfft2(refs).conj()
    shifts = torch.cat([torch.arange(9, device=device), torch.arange(52, 60, device=device)])
    stack = read_mrc_memmap(stack_path)
    chunks = []
    started = time.time()
    for offset in range(0, len(test), batch_size):
        ids = test[offset:offset + batch_size]
        path = out / f'pixel_block_{offset:05d}.npz'
        if path.exists():
            z = np.load(path)
            assert np.array_equal(z['ids'], ids)
            chunks.append(z['scores'])
            continue
        particles = torch.from_numpy(np.asarray(stack[ids]).copy()).to(device)
        particles = F.interpolate(particles[:, None], size=(size, size), mode='bilinear',
                                  align_corners=False, antialias=True)[:, 0]
        bg = mask < .05
        particles = ((particles - particles[:, bg].mean(1)[:, None, None]) /
                     particles[:, bg].std(1).clamp_min(1e-6)[:, None, None])
        best = torch.full((len(ids), len(templates)), -torch.inf, device=device)
        for angle in range(-180, 180, 12):
            moving = rotate_batch(particles, torch.full((len(ids),), float(angle), device=device))
            moving = normalise_for_match(band_limited_for_match(moving * mask, .30, .045), mask)
            corr = torch.fft.irfft2(torch.fft.rfft2(moving)[:, None] * fr[None], s=(size, size))
            values = corr.index_select(2, shifts).index_select(3, shifts).flatten(2).max(2).values
            if offset == 0 and angle == -180:
                n = min(2, len(ids))
                direct = restricted_best_shift(moving[:n], refs[:n], 8)[0]
                torch.testing.assert_close(values[torch.arange(n), torch.arange(n)], direct, atol=1e-6, rtol=1e-5)
            best = torch.maximum(best, values)
        scores = best.cpu().numpy()
        scores[:, counts == 0] = -np.inf
        assert np.isfinite(scores[:, counts > 0]).all()
        np.savez_compressed(path, ids=ids, scores=scores)
        chunks.append(scores)
        done = offset + len(ids)
        elapsed = time.time() - started
        emit(f'Pixel classification {done}/{len(test)} ({done / len(test):.1%}), elapsed={elapsed:.1f}s')
    return np.concatenate(chunks)

