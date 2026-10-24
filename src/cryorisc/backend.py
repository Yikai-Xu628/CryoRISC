from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .engine import circular_soft_mask, low_pass, rotate_batch, align_by_fourier_shift


def neutral_candidates(labels, budget, classes, rng):

    keys = rng.random((len(labels), classes))
    shortlist = np.argpartition(keys, budget-1, axis=1)[:, :budget]
    missing = ~np.any(shortlist == labels[:, None], axis=1)
    shortlist[missing, -1] = labels[missing]
    return shortlist.astype("i4"), np.zeros(shortlist.shape, "f4")


def prepare(config, arm, seed, particles, classes=200, candidate_budget=32):
    n, k = len(particles), classes
    rng = np.random.default_rng(seed)
    if arm == "B_no_network":


        labels = np.empty(n, "i4")
        labels[rng.permutation(n)] = np.arange(n, dtype="i4") % k
        anchors = np.array([rng.choice(np.flatnonzero(labels == c)) for c in range(k)])
        coarse_class, coarse_similarity = neutral_candidates(labels, candidate_budget, k, rng)
        embeddings = None
    else:
        with np.load(config["source"]) as source:
            labels = source["initial_kmeans_labels"].astype("i4")
            anchors = source["selected_anchor_ids"].astype("i8")
            embeddings = source["embeddings"].astype("f4")
        if not np.array_equal(labels[anchors], np.arange(k)):
            raise ValueError("Original feature anchor is outside its original class")
        similarity = embeddings @ embeddings[anchors].T
        coarse_class = np.argsort(-similarity, axis=1, kind="stable")[:, :candidate_budget].astype("i4")
        missing = ~np.any(coarse_class == labels[:, None], axis=1)
        coarse_class[missing, -1] = labels[missing]
        coarse_similarity = np.take_along_axis(similarity, coarse_class, 1).astype("f4")
    if labels.shape != (config["particles"],):
        raise ValueError("Source labels and raw stack have different particle counts")
    mask = circular_soft_mask(config["size"], config["radius"], config["edge"], torch.device("cpu"))
    values = torch.from_numpy(np.asarray(particles[anchors]).copy())
    bg = mask < .05
    values = (values-values[:, bg].mean(1)[:, None, None])/values[:, bg].std(1).clamp_min(1e-6)[:, None, None]
    templates = low_pass(values*mask, .08).numpy().astype("f4")
    return labels, anchors, embeddings, coarse_class, coarse_similarity, templates


def metadata(path, templates, classes, rotations, similarities=None):
    shape = classes.shape
    np.savez_compressed(path, templates=templates, candidate_classes=classes.astype("i4"),
        pair_rotations_deg=np.broadcast_to(rotations, shape).astype("f4"),
        candidate_similarities=np.zeros(shape, "f4") if similarities is None else similarities.astype("f4"),
        candidate_valid=np.ones(shape, bool), pair_reflections=np.zeros(shape, bool))


def full_angles(path, templates, classes, step, similarities=None):
    angles = np.arange(0, 360, step, dtype="f4")
    expanded = np.repeat(classes, len(angles), axis=1)
    expanded_s = None if similarities is None else np.repeat(similarities, len(angles), axis=1)
    metadata(path, templates, expanded, np.tile(angles, classes.shape[1])[None], expanded_s)


def fine_candidates(coarse_input, fit, labels, embeddings, anchors, fixed, class_count=200, budget=8):
    with np.load(coarse_input) as data:
        classes, rotations = data["candidate_classes"], data["pair_rotations_deg"]
    n = len(labels)
    mass = np.zeros((n, class_count), "f4")
    row = np.arange(n)[:, None]
    np.add.at(mass, (np.broadcast_to(row, classes.shape), classes), fit["pair_posterior"])
    chosen = labels[:, None] if fixed else np.argsort(-mass, axis=1, kind="stable")[:, :budget]
    base = np.empty(chosen.shape, "f4")
    for rank in range(chosen.shape[1]):
        best = np.argmax(np.where(classes == chosen[:, rank, None], fit["pair_posterior"], -1), axis=1)
        base[:, rank] = rotations[np.arange(n), best] - fit["pair_residual_angle"][np.arange(n), best]
    similarities = np.zeros(chosen.shape, "f4") if embeddings is None else np.sum(embeddings[:, None]*embeddings[anchors[chosen]], axis=2)
    return chosen, base, similarities.astype("f4")


def rescue_candidates(scores, assignment, budget=16):
    classes = np.argsort(-scores, axis=1, kind="stable")[:, :budget].copy()
    missing = ~np.any(classes == assignment[:, None], axis=1)
    classes[missing, -1] = assignment[missing]
    return classes.astype("i4")


def bandpass(x, size, low, high):
    fy, fx = torch.fft.fftfreq(size, device=x.device), torch.fft.rfftfreq(size, device=x.device)
    radius = torch.sqrt(fy[:, None]**2+fx[None, :]**2)
    weight = torch.sigmoid((radius-low)/.004)*torch.sigmoid((high-radius)/.012)
    return torch.fft.irfft2(torch.fft.rfft2(x)*weight, s=(size, size))


@torch.no_grad()
def final_average(config, key, inp, fit, particles, fixed_labels, folder, class_count=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(inp) as data:
        classes, rotations = data["candidate_classes"], data["pair_rotations_deg"]
    n, slots = len(particles), classes.shape[1]//60
    ids = classes.reshape(n, slots, 60)[:, :, 0]
    posterior = fit["pair_posterior"].reshape(n, slots, 60)
    mass = posterior.sum(2)
    topk = min(4, slots)
    top = np.argsort(-mass, axis=1, kind="stable")[:, :topk]
    row = np.arange(n)[:, None]
    labels = ids[row, top]
    best_angle = posterior[row, top].argmax(2)
    flat = top*60+best_angle
    angle = -rotations[row, flat]+fit["pair_residual_angle"][row, flat]
    dy, dx = fit["pair_shift_y"][row, flat], fit["pair_shift_x"][row, flat]
    if fixed_labels is not None and not np.all(labels == fixed_labels[:, None]):
        raise AssertionError("Final candidate crosses fixed class boundary")
    mask = circular_soft_mask(config["size"], config["radius"], config["edge"], device)
    bg = mask < .05
    low, high, temperature = (.018, .115, .055) if key == "10028" else (.015, .10, .05)
    templ = bandpass(torch.from_numpy(fit["templates"]).to(device), config["size"], low, high)*mask
    templ -= (templ*mask).sum((1, 2), keepdim=True)/mask.sum()
    templ /= torch.sqrt((templ.square()*mask).sum((1, 2), keepdim=True).clamp_min(1e-8))
    scores = np.empty(labels.shape, "f4")
    for start in range(0, n, 64):
        end = min(start+64, n)
        raw = torch.from_numpy(np.asarray(particles[start:end]).copy()).to(device)
        raw = (raw-raw[:, bg].mean(1)[:, None, None])/raw[:, bg].std(1).clamp_min(1e-6)[:, None, None]
        for rank in range(topk):
            image = rotate_batch(raw, torch.from_numpy(angle[start:end, rank].copy()).to(device))
            image = align_by_fourier_shift(image, torch.from_numpy(dy[start:end, rank].copy()).to(device), torch.from_numpy(dx[start:end, rank].copy()).to(device))
            image = bandpass(image, config["size"], low, high)*mask
            image -= (image*mask).sum((1, 2), keepdim=True)/mask.sum()
            image /= torch.sqrt((image.square()*mask).sum((1, 2), keepdim=True).clamp_min(1e-8))
            target = torch.from_numpy(labels[start:end, rank].copy()).to(device)
            scores[start:end, rank] = (image*templ[target]*mask).sum((1, 2)).cpu().numpy()
    weights = np.exp(np.clip((scores-scores.max(1, keepdims=True))/temperature, -40, 0))
    weights /= weights.sum(1, keepdims=True)

    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum(1, keepdims=True)
    if fixed_labels is not None:
        assert topk == 1 and np.all(weights == 1)
    sums = torch.zeros((class_count, config["size"], config["size"]), device=device)
    class_mass = torch.zeros(class_count, device=device)
    for start in range(0, n, 64):
        end = min(start+64, n)
        raw = torch.from_numpy(np.asarray(particles[start:end]).copy()).to(device)
        raw = (raw-raw[:, bg].mean(1)[:, None, None])/raw[:, bg].std(1).clamp_min(1e-6)[:, None, None]
        for rank in range(topk):
            image = rotate_batch(raw, torch.from_numpy(angle[start:end, rank].copy()).to(device))
            image = align_by_fourier_shift(image, torch.from_numpy(dy[start:end, rank].copy()).to(device), torch.from_numpy(dx[start:end, rank].copy()).to(device))
            target = torch.from_numpy(labels[start:end, rank].copy()).to(device)
            w = torch.from_numpy(weights[start:end, rank].copy()).to(device)
            sums.index_add_(0, target, image*w[:, None, None])
            class_mass.index_add_(0, target, w)
    averages = (sums/class_mass.clamp_min(1e-8)[:, None, None]).cpu().numpy().astype("f4")
    effective_counts = class_mass.cpu().numpy()
    if fixed_labels is not None:
        np.testing.assert_allclose(effective_counts, np.bincount(fixed_labels, minlength=class_count), atol=.02, rtol=0)
    np.savez_compressed(folder / "final_membership_and_poses.npz", classes=labels.astype("i4"),
        weights=weights, angles=angle, dy=dy, dx=dx, class_mass=effective_counts)
    return averages, effective_counts, labels[np.arange(n), weights.argmax(1)]


def unique_candidates(current, image_rank, feature_classes, feature_sims, budget=16):
    n = len(current)
    classes = np.empty((n, budget), dtype=np.int64)
    priors = np.empty((n, budget), dtype=np.float32)


    neutral = np.median(feature_sims, axis=1).astype(np.float32)
    for i in range(n):
        ordered = [int(current[i]), *map(int, image_rank[i, :8]), *map(int, feature_classes[i])]
        score_map = {int(c): float(s) for c, s in zip(feature_classes[i], feature_sims[i])}
        selected = []
        for c in ordered:
            if c not in selected:
                selected.append(c)
            if len(selected) == budget:
                break
        if len(selected) < budget:
            for c in image_rank[i, 8:]:
                c = int(c)
                if c not in selected:
                    selected.append(c)
                if len(selected) == budget:
                    break
        classes[i] = selected
        priors[i] = [score_map.get(c, float(neutral[i])) for c in selected]
    return classes, priors


def historical_fine_candidates(classes, posterior, rotations, similarity, budget=8):

    n = len(classes)
    mass = np.zeros((n, 200), 'f4')
    rows = np.arange(n)[:, None]
    np.add.at(mass, (np.broadcast_to(rows, classes.shape), classes), posterior)
    chosen = np.argpartition(-mass, budget-1, axis=1)[:, :budget]
    chosen_mass = np.take_along_axis(mass, chosen, 1)
    chosen = np.take_along_axis(chosen, np.argsort(-chosen_mass, axis=1), 1).astype('i4')
    base = np.empty(chosen.shape, 'f4')
    for rank in range(budget):
        slot = np.argmax(np.where(classes == chosen[:, rank, None], posterior, -1.), axis=1)
        base[:, rank] = rotations[np.arange(n), slot]
    return chosen, base, np.take_along_axis(similarity, chosen, 1).astype('f4')

