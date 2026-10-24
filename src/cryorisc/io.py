from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import struct
from .engine import read_mrc_memmap


def write_stack(path, images, apix):

    images = np.ascontiguousarray(images, dtype="<f4")
    nz, ny, nx = images.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", header, 28, nx, ny, 1)
    struct.pack_into("<3f", header, 40, nx * apix, ny * apix, apix)
    struct.pack_into("<3f", header, 52, 90., 90., 90.)
    struct.pack_into("<3i", header, 64, 1, 2, 3)
    struct.pack_into("<3f", header, 76, float(images.min()), float(images.max()), float(images.mean()))
    header[208:212] = b"MAP "
    header[212:216] = b"DA\0\0"
    struct.pack_into("<f", header, 216, float(images.std()))
    struct.pack_into("<i", header, 220, 1)
    label = b"CryoRISC native-size class averages"
    header[224:224 + len(label)] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        images.tofile(handle)

