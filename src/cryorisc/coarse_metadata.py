from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--angle-step", type=float, default=30.0)
    args = parser.parse_args()

    source = np.load(args.source)
    embeddings = source["embeddings"].astype(np.float32)
    anchor_ids = source["selected_anchor_ids"].astype(np.int64)
    class_embeddings = embeddings[anchor_ids]
    similarity = embeddings @ class_embeddings.T
    shortlist = np.argpartition(-similarity, args.classes - 1, axis=1)[:, : args.classes]
    shortlist_score = np.take_along_axis(similarity, shortlist, axis=1)
    order = np.argsort(-shortlist_score, axis=1)
    shortlist = np.take_along_axis(shortlist, order, axis=1).astype(np.int32)
    shortlist_score = np.take_along_axis(shortlist_score, order, axis=1).astype(np.float32)

    applied_angles = np.arange(-180.0, 180.0, args.angle_step, dtype=np.float32)
    angle_count = len(applied_angles)
    candidate_classes = np.repeat(shortlist, angle_count, axis=1)
    candidate_similarity = np.repeat(shortlist_score, angle_count, axis=1)
    pair_rotations = np.tile(-applied_angles, args.classes)[None].repeat(len(embeddings), axis=0)
    templates = np.load(args.template_result)["templates"].astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        templates=templates,
        embeddings=embeddings,
        selected_anchor_ids=anchor_ids,
        candidate_classes=candidate_classes,
        candidate_similarities=candidate_similarity,
        candidate_valid=np.ones_like(candidate_classes, dtype=bool),
        pair_rotations_deg=pair_rotations.astype(np.float32),
        pair_reflections=np.zeros_like(candidate_classes, dtype=bool),
    )
    report = {
        "particles": int(len(embeddings)),
        "total_class_centres": int(len(anchor_ids)),
        "shortlist_classes": args.classes,
        "angle_step_deg": args.angle_step,
        "angles_per_class": angle_count,
        "joint_hypotheses": int(candidate_classes.shape[1]),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
