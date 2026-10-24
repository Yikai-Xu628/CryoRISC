from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-input", type=Path, required=True)
    parser.add_argument("--coarse-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=8)
    args = parser.parse_args()

    coarse_input = np.load(args.coarse_input)
    coarse_result = np.load(args.coarse_result)
    candidate = coarse_input["candidate_classes"].astype(np.int32)
    posterior = coarse_result["pair_posterior"].astype(np.float32)
    rotations = coarse_input["pair_rotations_deg"].astype(np.float32)
    embeddings = coarse_input["embeddings"].astype(np.float32)
    anchor_ids = coarse_input["selected_anchor_ids"].astype(np.int64)
    class_embeddings = embeddings[anchor_ids]
    particle_count = len(candidate)
    total_classes = len(anchor_ids)

    class_mass = np.zeros((particle_count, total_classes), dtype=np.float32)
    rows = np.arange(particle_count)[:, None]
    np.add.at(class_mass, (np.broadcast_to(rows, candidate.shape), candidate), posterior)
    shortlist = np.argpartition(-class_mass, args.classes - 1, axis=1)[:, : args.classes]
    shortlist_mass = np.take_along_axis(class_mass, shortlist, axis=1)
    order = np.argsort(-shortlist_mass, axis=1)
    shortlist = np.take_along_axis(shortlist, order, axis=1).astype(np.int32)

    best_rotation = np.empty((particle_count, args.classes), dtype=np.float32)
    for rank in range(args.classes):
        class_id = shortlist[:, rank]
        belongs = candidate == class_id[:, None]
        best_slot = np.argmax(np.where(belongs, posterior, -1.0), axis=1)
        best_rotation[:, rank] = rotations[np.arange(particle_count), best_slot]

    learned_similarity = np.sum(
        embeddings[:, None, :] * class_embeddings[shortlist], axis=2
    ).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        templates=coarse_result["templates"].astype(np.float32),
        embeddings=embeddings,
        selected_anchor_ids=anchor_ids,
        candidate_classes=shortlist,
        candidate_similarities=learned_similarity,
        candidate_valid=np.ones_like(shortlist, dtype=bool),
        pair_rotations_deg=best_rotation,
        pair_reflections=np.zeros_like(shortlist, dtype=bool),
    )
    report = {
        "particles": particle_count,
        "fine_candidate_classes": args.classes,
        "source_joint_hypotheses": int(candidate.shape[1]),
        "coarse_mean_retained_class_mass": float(
            np.take_along_axis(class_mass, shortlist, axis=1).sum(1).mean()
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
