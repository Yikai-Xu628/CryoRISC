from __future__ import annotations

import argparse
import json
import time
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_mrc_memmap(path: Path) -> np.memmap:

    with path.open("rb") as handle:
        header = handle.read(1024)
    nx, ny, nz, mode = struct.unpack_from("<4i", header, 0)
    nsymbt = struct.unpack_from("<i", header, 92)[0]
    if mode != 2:
        raise ValueError(f"Only float32 MRC mode 2 is supported, got mode={mode}: {path}")
    return np.memmap(
        path, mode="r", dtype="<f4", offset=1024 + nsymbt, shape=(nz, ny, nx)
    )


def circular_soft_mask(size: int, radius: float, edge: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij"
    )
    rr = torch.sqrt((xx - (size - 1) / 2.0) ** 2 + (yy - (size - 1) / 2.0) ** 2)
    mask = torch.ones_like(rr, dtype=torch.float32)
    transition = (rr - radius) / max(edge, 1e-6)
    mask = torch.where(rr <= radius, mask, 0.5 * (1.0 + torch.cos(math.pi * transition.clamp(0, 1))))
    return torch.where(rr >= radius + edge, torch.zeros_like(mask), mask)


def rotate_batch(images: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(angle_deg)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    affine = torch.zeros((images.shape[0], 2, 3), device=images.device, dtype=images.dtype)
    affine[:, 0, 0] = cosine
    affine[:, 0, 1] = -sine
    affine[:, 1, 0] = sine
    affine[:, 1, 1] = cosine
    grid = F.affine_grid(affine, images[:, None].shape, align_corners=False)
    return F.grid_sample(
        images[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[:, 0]


def normalise_for_match(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    mean = (images * mask).sum((-2, -1), keepdim=True) / denom
    centred = (images - mean) * mask
    return centred / torch.sqrt((centred.square()).sum((-2, -1), keepdim=True).clamp_min(1e-8))


def low_pass(images: torch.Tensor, cutoff: float) -> torch.Tensor:
    n = images.shape[-1]
    fy = torch.fft.fftfreq(n, device=images.device)[:, None]
    fx = torch.fft.rfftfreq(n, device=images.device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    filt = 1.0 / (1.0 + (radius / max(cutoff, 1e-4)) ** 8)
    return torch.fft.irfft2(torch.fft.rfft2(images) * filt, s=(n, n))


def band_limited_for_match(
    images: torch.Tensor, cutoff: float, min_frequency: float
) -> torch.Tensor:

    n = images.shape[-1]
    fy = torch.fft.fftfreq(n, device=images.device)[:, None]
    fx = torch.fft.rfftfreq(n, device=images.device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    low = 1.0 / (1.0 + (radius / max(cutoff, 1e-4)) ** 8)
    if min_frequency > 0:
        high = 1.0 - 1.0 / (1.0 + (radius / min_frequency) ** 8)
        filt = low * high
    else:
        filt = low
    return torch.fft.irfft2(torch.fft.rfft2(images) * filt, s=(n, n))


def restricted_best_shift(
    moving: torch.Tensor,
    references: torch.Tensor,
    max_shift: int,
    subpixel: bool = False,
    prior_strength: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = moving.shape[-1]
    fm = torch.fft.rfft2(moving)
    fr = torch.fft.rfft2(references)
    corr = torch.fft.irfft2(torch.conj(fr) * fm, s=(n, n))
    indices = torch.cat(
        (torch.arange(0, max_shift + 1, device=moving.device),
         torch.arange(n - max_shift, n, device=moving.device))
    )
    local = corr.index_select(1, indices).index_select(2, indices)
    if prior_strength > 0:
        signed = torch.where(indices <= max_shift, indices, indices - n).to(moving.dtype)
        radius2 = signed[:, None].square() + signed[None, :].square()


        local = local - prior_strength * radius2[None] / float(max_shift * max_shift)
    flat = local.flatten(1)
    score, best = flat.max(1)
    iy = indices[best // indices.numel()]
    ix = indices[best % indices.numel()]
    dy = torch.where(iy <= max_shift, iy, iy - n).to(moving.dtype)
    dx = torch.where(ix <= max_shift, ix, ix - n).to(moving.dtype)
    if subpixel:
        rows = torch.arange(moving.shape[0], device=moving.device)

        def parabolic_delta(left: torch.Tensor, mid: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            denom = left - 2.0 * mid + right
            delta = torch.where(
                denom.abs() > 1e-8,
                0.5 * (left - right) / denom,
                torch.zeros_like(mid),
            )
            return delta.clamp(-0.5, 0.5)

        peak = corr[rows, iy, ix]
        delta_y = parabolic_delta(
            corr[rows, (iy - 1) % n, ix], peak, corr[rows, (iy + 1) % n, ix]
        )
        delta_x = parabolic_delta(
            corr[rows, iy, (ix - 1) % n], peak, corr[rows, iy, (ix + 1) % n]
        )

        delta_y = torch.where(dy.abs() < max_shift, delta_y, torch.zeros_like(delta_y))
        delta_x = torch.where(dx.abs() < max_shift, delta_x, torch.zeros_like(delta_x))
        dy = dy + delta_y
        dx = dx + delta_x
    return score, dy, dx


def align_by_integer_shift(images: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(images)
    shifts = torch.stack((dy, dx), dim=1)
    for shift in torch.unique(shifts, dim=0):
        use = torch.all(shifts == shift[None], dim=1)

        output[use] = torch.roll(
            images[use], shifts=(-int(shift[0]), -int(shift[1])), dims=(-2, -1)
        )
    return output


def align_by_fourier_shift(images: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:

    n = images.shape[-1]
    fy = torch.fft.fftfreq(n, device=images.device, dtype=images.dtype)[:, None]
    fx = torch.fft.rfftfreq(n, device=images.device, dtype=images.dtype)[None, :]
    phase_argument = dy[:, None, None] * fy + dx[:, None, None] * fx
    phase = torch.exp((2j * math.pi) * phase_argument)
    return torch.fft.irfft2(torch.fft.rfft2(images) * phase, s=(n, n))


def bounded_class_prior(
    mass: torch.Tensor,
    floor_fraction: float,
    maximum_multiple: float,
) -> torch.Tensor:

    classes = mass.numel()
    floor = max(0.0, floor_fraction) / classes
    cap = (
        maximum_multiple / classes
        if maximum_multiple > 0 else 1.0
    )
    if floor * classes >= 1.0 or cap * classes < 1.0:
        raise ValueError("Incompatible class-prior floor/cap")
    weights = mass.clamp_min(0)
    if float(weights.sum()) <= 0:
        return torch.full_like(weights, 1.0 / classes)
    prior = torch.full_like(weights, floor)
    remaining = 1.0 - floor * classes
    room = torch.full_like(weights, cap - floor)
    active = room > 1e-12


    for _ in range(classes + 2):
        if remaining <= 1e-8 or not bool(active.any()):
            break
        active_weights = weights * active
        if float(active_weights.sum()) <= 0:
            active_weights = active.to(weights.dtype)
        proposal = remaining * active_weights / active_weights.sum()
        addition = torch.minimum(proposal, room)
        prior += addition
        room -= addition
        remaining -= float(addition.sum())
        active = room > 1e-8
    if remaining > 1e-6:
        prior += remaining / classes
    return prior / prior.sum()


def select_hard_hypothesis(
    posterior: torch.Tensor,
    candidate: torch.Tensor,
    valid_candidate: torch.Tensor,
    classes: int,
    aggregate_duplicate_candidates: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    if aggregate_duplicate_candidates:
        class_posterior = torch.zeros(
            posterior.shape[0], classes, dtype=posterior.dtype, device=posterior.device
        )
        class_posterior.scatter_add_(
            1, candidate, posterior * valid_candidate.to(posterior.dtype)
        )
        class_top_values, class_top_indices = torch.topk(
            class_posterior, k=min(2, classes), dim=1
        )
        maximum = class_top_values[:, 0]
        hard_class = class_top_indices[:, 0]
        margin = (
            maximum - class_top_values[:, 1]
            if classes > 1 else torch.ones_like(maximum)
        )
        winning_pose = valid_candidate & (candidate == hard_class[:, None])
        maximum_index = posterior.masked_fill(~winning_pose, -1.0).argmax(1)
        entropy_posterior = class_posterior
    else:
        top_values, top_indices = torch.topk(
            posterior, k=min(2, posterior.shape[1]), dim=1
        )
        maximum = top_values[:, 0]
        maximum_index = top_indices[:, 0]
        margin = (
            maximum - top_values[:, 1]
            if posterior.shape[1] > 1 else torch.ones_like(maximum)
        )
        hard_class = candidate.gather(1, maximum_index[:, None]).squeeze(1)
        entropy_posterior = posterior
    return maximum, maximum_index, margin, hard_class, entropy_posterior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--fixed-labels-npy", type=Path, default=None,
        help="Require every valid candidate to equal this immutable per-particle class label.",
    )
    parser.add_argument(
        "--export-unfiltered-averages", action="store_true",
        help="Save final aligned-particle sums divided by update mass before masking/filtering/momentum.",
    )
    parser.add_argument(
        "--images-npy", type=Path, default=None,
        help="Optional N x H x W float32 NPY stack. This keeps large real-data images out of the metadata NPZ.",
    )
    parser.add_argument(
        "--images-mrcs", type=Path, default=None,
        help="Optional float32 MRCS stack, memory-mapped directly to avoid a duplicate NPY cache.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-result", type=Path, default=None,
        help="Use templates and per-candidate angles from an earlier coarse refinement result.",
    )
    parser.add_argument(
        "--template-result", type=Path, default=None,
        help="Replace only the starting templates from an earlier result while preserving input candidate angles.",
    )
    parser.add_argument(
        "--cpu-images", action="store_true",
        help="Keep the full image stack in CPU memory and transfer/normalise one batch at a time.",
    )
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument(
        "--shift-prior-strength", type=float, default=0.0,
        help="Quadratic zero-centred shift penalty at one max-shift radius.",
    )
    parser.add_argument("--residual-angle", type=float, default=2.0)
    parser.add_argument("--angle-step", type=float, default=0.0)
    parser.add_argument(
        "--marginalize-angles",
        action="store_true",
        help="Normalize one joint posterior over candidate class and residual-angle hypotheses.",
    )
    parser.add_argument(
        "--aggregate-duplicate-candidates",
        action="store_true",
        help=(
            "Sum posterior mass over repeated candidate columns of the same class "
            "before selecting the final class; required for an expanded full-angle grid."
        ),
    )
    parser.add_argument("--subpixel", action="store_true")
    parser.add_argument("--use-precomputed-reflection", action="store_true")
    parser.add_argument("--min-match-frequency", type=float, default=0.0)
    parser.add_argument("--max-cutoff", type=float, default=0.34)
    parser.add_argument("--temperature", type=float, default=0.035)
    parser.add_argument("--embedding-weight", type=float, default=0.35)
    parser.add_argument(
        "--class-prior-power",
        type=float,
        default=1.0,
        help="0 disables rich-get-richer class priors; 1 reproduces the original update.",
    )
    parser.add_argument(
        "--prior-floor-fraction", type=float, default=0.20,
        help="Class-prior floor as a fraction of 1/K. Use a small value to permit nearly empty classes.",
    )
    parser.add_argument(
        "--max-class-prior-multiple", type=float, default=0.0,
        help="Optional upper bound relative to uniform 1/K; zero disables the cap.",
    )
    parser.add_argument(
        "--template-update-min-maxprob", type=float, default=0.0,
        help="Only particles above this maximum posterior may update class templates.",
    )
    parser.add_argument(
        "--template-update-min-margin", type=float, default=0.0,
        help="Minimum top-1 minus top-2 posterior margin for template updates.",
    )
    parser.add_argument(
        "--template-update-relax", type=float, default=0.0,
        help="Decrease the maximum-posterior update threshold by this amount each iteration.",
    )
    parser.add_argument(
        "--template-update-margin-relax", type=float, default=0.0,
        help="Decrease the posterior-margin update threshold by this amount each iteration.",
    )
    parser.add_argument(
        "--hard-core-template-updates", action="store_true",
        help="A confident particle updates only its best class; assignment posteriors remain soft.",
    )
    parser.add_argument(
        "--prior-from-template-updates", action="store_true",
        help="Estimate class priors from confident template-update mass instead of all posterior mass.",
    )
    parser.add_argument(
        "--core-only-template-iterations", type=int, default=0,
        help="For the first N iterations, update templates only from input core_membership particles.",
    )
    parser.add_argument(
        "--lock-core-update-class", action="store_true",
        help="During core-only iterations, force each core particle to update its graph-core class.",
    )
    parser.add_argument("--template-momentum", type=float, default=0.20)
    parser.add_argument("--mask-radius", type=float, default=50.0)
    parser.add_argument("--mask-edge", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.input)
    if args.images_npy is not None and args.images_mrcs is not None:
        raise ValueError("Use only one of --images-npy and --images-mrcs")
    if args.images_npy is not None:
        image_source = np.load(args.images_npy, mmap_mode="r")
    elif args.images_mrcs is not None:
        image_source = read_mrc_memmap(args.images_mrcs)
    else:
        image_source = raw["images"]
    n_all = image_source.shape[0]
    n = n_all if args.limit <= 0 else min(args.limit, n_all)
    images_np = image_source[:n]
    if images_np.dtype != np.float32:
        images_np = images_np.astype(np.float32)
    classes_np = raw["candidate_classes"][:n].astype(np.int64)
    valid_np = (
        raw["candidate_valid"][:n].astype(bool)
        if "candidate_valid" in raw.files
        else np.ones_like(classes_np, dtype=bool)
    )
    fixed_labels = None
    if args.fixed_labels_npy is not None:
        stored_labels = np.load(args.fixed_labels_npy, allow_pickle=False)
        if stored_labels.shape != (n_all,) or not np.issubdtype(stored_labels.dtype, np.integer):
            raise ValueError("Fixed labels must contain one integer label per source particle")
        fixed_labels = stored_labels[:n].astype(np.int64)
        if not np.all(valid_np.any(axis=1)):
            raise ValueError("A fixed-label particle has no valid pose candidate")
        if np.any(valid_np & (classes_np != fixed_labels[:, None])):
            raise ValueError("Fixed-label run contains a candidate from a different class")
    similarities_np = raw["candidate_similarities"][:n].astype(np.float32)
    rotations_np = raw["pair_rotations_deg"][:n].astype(np.float32)
    reflections_np = (
        raw["pair_reflections"][:n].astype(bool)
        if "pair_reflections" in raw.files
        else np.zeros_like(valid_np, dtype=bool)
    )
    core_membership_np = (
        raw["core_membership"][:n].astype(np.int64)
        if "core_membership" in raw.files
        else np.full(n, -1, dtype=np.int64)
    )
    has_ground_truth = "viewdirs" in raw.files
    if has_ground_truth:
        viewdirs_np = raw["viewdirs"][:n].astype(np.float32)
        has_ground_truth = bool(np.all(np.linalg.norm(viewdirs_np, axis=1) > 0.5))
    if has_ground_truth:
        anchor_dirs_np = raw["viewdirs"][raw["selected_anchor_ids"].astype(np.int64)].astype(np.float32)
    else:
        viewdirs_np = None
        anchor_dirs_np = None

    if args.resume_result is not None:
        resume = np.load(args.resume_result)
        templates_np = resume["templates"].astype(np.float32)
        if "pair_residual_angle" not in resume.files:
            raise ValueError("Resume result has no pair_residual_angle")


        rotations_np = -resume["pair_residual_angle"][:n].astype(np.float32)
    else:
        templates_np = raw["templates"].astype(np.float32)
    if args.template_result is not None:
        if args.resume_result is not None:
            raise ValueError("--template-result and --resume-result are mutually exclusive")
        templates_np = np.load(args.template_result)["templates"].astype(np.float32)
    templates = torch.from_numpy(templates_np).to(device)
    k = templates.shape[0]
    if fixed_labels is not None and np.any((fixed_labels < 0) | (fixed_labels >= k)):
        raise ValueError("Fixed label outside template class range")
    topm = classes_np.shape[1]
    size = templates.shape[-1]
    mask = circular_soft_mask(size, args.mask_radius, args.mask_edge, device)
    background = mask < 0.05


    images = torch.from_numpy(images_np)
    if not args.cpu_images:
        images = images.to(device)
        bg_mean = images[:, background].mean(1, keepdim=True)
        bg_std = images[:, background].std(1, keepdim=True).clamp_min(1e-6)
        images = (images - bg_mean[:, None]) / bg_std[:, None]
    classes = torch.from_numpy(classes_np).to(device)
    candidate_valid = torch.from_numpy(valid_np).to(device)
    similarities = torch.from_numpy(similarities_np).to(device)
    base_angles = -torch.from_numpy(rotations_np).to(device)
    pair_reflections = torch.from_numpy(reflections_np).to(device)
    class_prior = torch.full((k,), 1.0 / k, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    final_assignment = np.empty(n, dtype=np.int32)
    final_probability = np.empty(n, dtype=np.float32)
    final_dy = np.empty(n, dtype=np.float32)
    final_dx = np.empty(n, dtype=np.float32)
    final_residual = np.empty(n, dtype=np.float32)
    final_reflection = np.empty(n, dtype=bool)
    final_pair_posterior = np.zeros((n, topm), dtype=np.float32)
    final_pair_dy = np.zeros((n, topm), dtype=np.float32)
    final_pair_dx = np.zeros((n, topm), dtype=np.float32)
    final_pair_residual = np.zeros((n, topm), dtype=np.float32)
    final_best_candidate_index = np.empty(n, dtype=np.int32)

    print(f"device={device} particles={n} classes={k} candidates={topm}", flush=True)
    for iteration in range(args.iterations):
        iteration_started = time.perf_counter()
        cutoff = min(args.max_cutoff, 0.20 + 0.035 * iteration)
        update_min_probability = max(
            0.0, args.template_update_min_maxprob - iteration * args.template_update_relax
        )
        update_min_margin = max(
            0.0, args.template_update_min_margin
            - iteration * args.template_update_margin_relax
        )
        match_templates = normalise_for_match(
            band_limited_for_match(templates * mask, cutoff, args.min_match_frequency), mask
        )
        sums = torch.zeros_like(templates)
        occupancy = torch.zeros(k, device=device)
        template_occupancy = torch.zeros(k, device=device)
        entropy_sum = 0.0
        maxprob_sum = 0.0
        template_update_particles = 0.0
        weighted_purity_sum = 0.0

        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            b = stop - start
            batch_images = images[start:stop].to(device)
            if args.cpu_images:
                bg_mean = batch_images[:, background].mean(1, keepdim=True)
                bg_std = batch_images[:, background].std(1, keepdim=True).clamp_min(1e-6)
                batch_images = (batch_images - bg_mean[:, None]) / bg_std[:, None]
            candidate = classes[start:stop]
            valid_candidate = candidate_valid[start:stop]
            angle0 = base_angles[start:stop]
            emb_sim = similarities[start:stop]
            batch_core_membership = torch.from_numpy(
                core_membership_np[start:stop]
            ).to(device)


            if args.angle_step > 0:
                residuals = torch.arange(
                    -args.residual_angle,
                    args.residual_angle + 0.5 * args.angle_step,
                    args.angle_step,
                    device=device,
                )
            else:
                residuals = torch.tensor(
                    [-args.residual_angle, 0.0, args.residual_angle], device=device
                )
            best_score = torch.full((b, topm), -1e9, device=device)
            best_dy = torch.zeros((b, topm), dtype=torch.float32, device=device)
            best_dx = torch.zeros((b, topm), dtype=torch.float32, device=device)
            best_residual = torch.zeros((b, topm), dtype=torch.float32, device=device)
            best_rotated = torch.empty((b, topm, size, size), device=device)
            refs = match_templates[candidate.reshape(-1)]
            repeated = batch_images[:, None].expand(-1, topm, -1, -1).reshape(-1, size, size)
            if args.use_precomputed_reflection:
                reflected = pair_reflections[start:stop].reshape(-1)
                repeated = torch.where(
                    reflected[:, None, None], torch.flip(repeated, dims=(-2,)), repeated
                )

            joint_scores: list[torch.Tensor] = []
            joint_dy: list[torch.Tensor] = []
            joint_dx: list[torch.Tensor] = []
            for residual in residuals:
                rotated = rotate_batch(repeated, (angle0 + residual).reshape(-1))
                moving = normalise_for_match(
                    band_limited_for_match(rotated * mask, cutoff, args.min_match_frequency), mask
                )
                score, dy, dx = restricted_best_shift(
                    moving, refs, args.max_shift, subpixel=args.subpixel,
                    prior_strength=args.shift_prior_strength,
                )
                score = score.reshape(b, topm)
                if args.marginalize_angles:
                    joint_scores.append(score)
                    joint_dy.append(dy.reshape(b, topm))
                    joint_dx.append(dx.reshape(b, topm))
                update = score > best_score
                best_score = torch.where(update, score, best_score)
                best_dy = torch.where(update, dy.reshape(b, topm), best_dy)
                best_dx = torch.where(update, dx.reshape(b, topm), best_dx)
                best_residual = torch.where(
                    update, torch.full_like(best_residual, float(residual)), best_residual
                )
                update4 = update[:, :, None, None]
                best_rotated = torch.where(update4, rotated.reshape(b, topm, size, size), best_rotated)

            embedding_prior = args.embedding_weight * emb_sim / 0.05
            class_log_prior = args.class_prior_power * torch.log(
                class_prior[candidate].clamp_min(1e-8)
            )
            if args.marginalize_angles:
                score_cube = torch.stack(joint_scores, dim=2)
                dy_cube = torch.stack(joint_dy, dim=2)
                dx_cube = torch.stack(joint_dx, dim=2)
                joint_logits = score_cube / args.temperature
                joint_logits += embedding_prior[:, :, None]
                joint_logits += class_log_prior[:, :, None]
                joint_logits = joint_logits.masked_fill(~valid_candidate[:, :, None], -1e9)
                joint_posterior = torch.softmax(joint_logits.flatten(1), dim=1).reshape_as(joint_logits)
                posterior = joint_posterior.sum(2)

                maximum, maximum_index, margin, hard_class, entropy_posterior = (
                    select_hard_hypothesis(
                        posterior, candidate, valid_candidate, k,
                        args.aggregate_duplicate_candidates,
                    )
                )
                update_gate = (
                    (maximum >= update_min_probability)
                    & (margin >= update_min_margin)
                )
                update_index = maximum_index
                if iteration < args.core_only_template_iterations:
                    core_candidate = (
                        candidate == batch_core_membership[:, None]
                    ) & valid_candidate & (batch_core_membership[:, None] >= 0)
                    core_present = core_candidate.any(1)
                    update_gate = core_present
                    if args.lock_core_update_class:
                        update_index = posterior.masked_fill(
                            ~core_candidate, -1.0
                        ).argmax(1)
                update_posterior = posterior * update_gate[:, None]
                if args.hard_core_template_updates:
                    update_posterior = torch.zeros_like(posterior).scatter(
                        1, update_index[:, None], update_gate[:, None].to(posterior.dtype)
                    )


                flat_classes = candidate.reshape(-1)
                occupancy.index_add_(0, flat_classes, posterior.reshape(-1))
                template_occupancy.index_add_(
                    0, flat_classes, update_posterior.reshape(-1)
                )
                conditional = joint_posterior / posterior[:, :, None].clamp_min(1e-12)
                update_joint_posterior = conditional * update_posterior[:, :, None]
                for angle_index, residual in enumerate(residuals):
                    rotated = rotate_batch(repeated, (angle0 + residual).reshape(-1))
                    if args.subpixel:
                        aligned_angle = align_by_fourier_shift(
                            rotated,
                            dy_cube[:, :, angle_index].reshape(-1),
                            dx_cube[:, :, angle_index].reshape(-1),
                        ).reshape(b, topm, size, size)
                    else:
                        aligned_angle = align_by_integer_shift(
                            rotated,
                            dy_cube[:, :, angle_index].round().long().reshape(-1),
                            dx_cube[:, :, angle_index].round().long().reshape(-1),
                        ).reshape(b, topm, size, size)
                    angle_weight = update_joint_posterior[:, :, angle_index]
                    sums.index_add_(
                        0,
                        flat_classes,
                        (aligned_angle * angle_weight[:, :, None, None]).reshape(-1, size, size),
                    )


                best_dy = (conditional * dy_cube).sum(2)
                best_dx = (conditional * dx_cube).sum(2)
                best_residual = (
                    conditional * residuals[None, None, :]
                ).sum(2)
            else:
                logits = best_score / args.temperature
                logits += embedding_prior
                logits += class_log_prior
                logits = logits.masked_fill(~valid_candidate, -1e9)
                posterior = torch.softmax(logits, dim=1)
                maximum, maximum_index, margin, hard_class, entropy_posterior = (
                    select_hard_hypothesis(
                        posterior, candidate, valid_candidate, k,
                        args.aggregate_duplicate_candidates,
                    )
                )
                update_gate = (
                    (maximum >= update_min_probability)
                    & (margin >= update_min_margin)
                )
                update_index = maximum_index
                if iteration < args.core_only_template_iterations:
                    core_candidate = (
                        candidate == batch_core_membership[:, None]
                    ) & valid_candidate & (batch_core_membership[:, None] >= 0)
                    core_present = core_candidate.any(1)
                    update_gate = core_present
                    if args.lock_core_update_class:
                        update_index = posterior.masked_fill(
                            ~core_candidate, -1.0
                        ).argmax(1)
                update_posterior = posterior * update_gate[:, None]
                if args.hard_core_template_updates:
                    update_posterior = torch.zeros_like(posterior).scatter(
                        1, update_index[:, None], update_gate[:, None].to(posterior.dtype)
                    )
                if args.subpixel:
                    aligned = align_by_fourier_shift(
                        best_rotated.reshape(-1, size, size),
                        best_dy.reshape(-1), best_dx.reshape(-1),
                    ).reshape(b, topm, size, size)
                else:
                    aligned = align_by_integer_shift(
                        best_rotated.reshape(-1, size, size),
                        best_dy.round().long().reshape(-1), best_dx.round().long().reshape(-1),
                    ).reshape(b, topm, size, size)

                flat_classes = candidate.reshape(-1)
                flat_weights = posterior.reshape(-1)
                occupancy.index_add_(0, flat_classes, flat_weights)
                template_occupancy.index_add_(
                    0, flat_classes, update_posterior.reshape(-1)
                )
                sums.index_add_(
                    0, flat_classes,
                    (aligned * update_posterior[:, :, None, None]).reshape(-1, size, size),
                )

            entropy_sum += float((-(entropy_posterior * entropy_posterior.clamp_min(1e-8).log()).sum(1)).sum())
            maxprob_sum += float(maximum.sum())
            template_update_particles += float(update_gate.sum())
            hard_dy = best_dy.gather(1, maximum_index[:, None]).squeeze(1)
            hard_dx = best_dx.gather(1, maximum_index[:, None]).squeeze(1)
            hard_residual = best_residual.gather(1, maximum_index[:, None]).squeeze(1)
            hard_reflection = pair_reflections[start:stop].gather(
                1, maximum_index[:, None]
            ).squeeze(1)
            final_assignment[start:stop] = hard_class.detach().cpu().numpy().astype(np.int32)
            final_probability[start:stop] = maximum.detach().cpu().numpy().astype(np.float32)
            final_dy[start:stop] = hard_dy.detach().cpu().numpy().astype(np.float32)
            final_dx[start:stop] = hard_dx.detach().cpu().numpy().astype(np.float32)
            final_residual[start:stop] = hard_residual.detach().cpu().numpy().astype(np.float32)
            final_reflection[start:stop] = hard_reflection.detach().cpu().numpy().astype(bool)
            final_pair_posterior[start:stop] = posterior.detach().cpu().numpy().astype(np.float32)
            final_pair_dy[start:stop] = best_dy.detach().cpu().numpy().astype(np.float32)
            final_pair_dx[start:stop] = best_dx.detach().cpu().numpy().astype(np.float32)
            final_pair_residual[start:stop] = best_residual.detach().cpu().numpy().astype(np.float32)
            final_best_candidate_index[start:stop] = maximum_index.detach().cpu().numpy().astype(np.int32)

            if has_ground_truth:
                dirs = torch.from_numpy(viewdirs_np[start:stop]).to(device)
                anchor_dirs = torch.from_numpy(anchor_dirs_np).to(device)[candidate]
                dots = (dirs[:, None] * anchor_dirs).sum(2).abs().clamp(0, 1)
                angular = torch.rad2deg(torch.acos(dots))
                weighted_purity_sum += float((posterior * (angular <= 8).float()).sum())

            if stop == n or stop % 500 < args.batch_size:
                elapsed = time.perf_counter() - iteration_started
                rate = stop / max(elapsed, 1e-6)
                remaining = (n - stop) / max(rate, 1e-6)
                print(
                    f"iteration={iteration + 1}/{args.iterations} "
                    f"particles={stop}/{n} rate={rate:.1f}/s eta={remaining / 60:.1f}min",
                    flush=True,
                )

        if fixed_labels is not None and not np.array_equal(final_assignment, fixed_labels):
            raise AssertionError("Fixed class membership changed during refinement")
        if args.export_unfiltered_averages:
            unfiltered_averages_np = (
                sums / template_occupancy.clamp_min(1e-8)[:, None, None]
            ).detach().cpu().numpy().astype(np.float32)
        valid = template_occupancy >= 2.0
        updated = templates.clone()
        updated[valid] = sums[valid] / template_occupancy[valid, None, None]
        updated = low_pass(updated * mask, cutoff)
        templates = args.template_momentum * templates + (1.0 - args.template_momentum) * updated
        prior_mass = template_occupancy if args.prior_from_template_updates else occupancy
        if float(prior_mass.sum()) <= 0:
            prior_mass = occupancy
        class_prior = bounded_class_prior(
            prior_mass,
            args.prior_floor_fraction,
            args.max_class_prior_multiple,
        )

        active = occupancy[occupancy > 0]
        record = {
            "iteration": iteration + 1,
            "cutoff_cycles_per_pixel": cutoff,
            "mean_max_posterior": maxprob_sum / n,
            "mean_entropy": entropy_sum / n,
            "template_update_min_probability": update_min_probability,
            "template_update_min_margin": update_min_margin,
            "template_update_particle_fraction": template_update_particles / n,
            "core_only_template_iteration": bool(
                iteration < args.core_only_template_iterations
            ),
            "occupancy_min": float(active.min()) if active.numel() else 0.0,
            "occupancy_median": float(active.median()) if active.numel() else 0.0,
            "occupancy_max": float(active.max()) if active.numel() else 0.0,
            "active_classes": int(active.numel()),
            "template_active_classes": int((template_occupancy > 0).sum()),
            "template_occupancy_median": float(
                template_occupancy[template_occupancy > 0].median()
            ) if bool((template_occupancy > 0).any()) else 0.0,
            "class_prior_min": float(class_prior.min()),
            "class_prior_max": float(class_prior.max()),
        }
        if has_ground_truth:
            record["weighted_purity8"] = weighted_purity_sum / n
        if fixed_labels is not None:
            record["changed_fixed_labels"] = 0
        history.append(record)
        print(json.dumps(record), flush=True)

    templates_np = templates.detach().cpu().numpy().astype(np.float32)
    result_npz = args.output_dir / "soft_refine_result.npz"
    extra_result = {}
    if args.export_unfiltered_averages:
        extra_result["unfiltered_averages"] = unfiltered_averages_np
    np.savez_compressed(
        result_npz,
        templates=templates_np,
        occupancy=occupancy.detach().cpu().numpy().astype(np.float32),
        template_occupancy=template_occupancy.detach().cpu().numpy().astype(np.float32),
        class_prior=class_prior.detach().cpu().numpy().astype(np.float32),
        assignment=final_assignment,
        assignment_probability=final_probability,
        best_shift_y=final_dy,
        best_shift_x=final_dx,
        best_residual_angle=final_residual,
        best_reflection=final_reflection,
        pair_posterior=final_pair_posterior,
        pair_shift_y=final_pair_dy,
        pair_shift_x=final_pair_dx,
        pair_residual_angle=final_pair_residual,
        best_candidate_index=final_best_candidate_index,
        history_json=np.asarray(json.dumps(history)),
        **extra_result,
    )
    with (args.output_dir / "soft_refine_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"saved={result_npz}", flush=True)


if __name__ == "__main__":
    main()
