#!/usr/bin/env python3
"""Validate all development evaluation arrays before metric computation."""

from __future__ import annotations

import csv
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


NUM_ENVS = 64
NUM_STEPS = 3000
SCENARIO_SEED = 101
ITERATIONS = tuple(range(0, 3000, 100)) + (2999,)

EXPECTED_SHAPES = {
    "base_angular_velocity.npy": (NUM_ENVS, 3, NUM_STEPS),
    "base_position.npy": (NUM_ENVS, 3, NUM_STEPS),
    "base_quaternion.npy": (NUM_ENVS, 4, NUM_STEPS),
    "base_tracked_velocity.npy": (NUM_ENVS, 3, NUM_STEPS),
    "command.npy": (NUM_ENVS, 3, NUM_STEPS),
    "death_status.npy": (NUM_ENVS, 1, NUM_STEPS),
    "energy.npy": (NUM_ENVS, 1, NUM_STEPS),
    "feet_contacts.npy": (NUM_ENVS, 4, NUM_STEPS),
    "feet_forces.npy": (NUM_ENVS, 4, 3, NUM_STEPS),
    "feet_position.npy": (NUM_ENVS, 4, 3, NUM_STEPS),
    "hip_position_z.npy": (NUM_ENVS, NUM_STEPS),
    "joint_acceleration.npy": (NUM_ENVS, 12, NUM_STEPS),
    "joint_position.npy": (NUM_ENVS, 12, NUM_STEPS),
    "robot_mass.npy": (),
    "survival_mask.npy": (NUM_ENVS, 1, NUM_STEPS),
    "survival_time.npy": (NUM_ENVS,),
}


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


ROOT = repository_root()
BASE = ROOT / "analysis_outputs" / "part_b_from_scratch"
MANIFEST = BASE / "formal_audit" / "formal_run_manifest.tsv"
SWEEP_MARKER = BASE / "formal_audit" / "DEV_CHECKPOINT_SWEEP_PASS"
OUTPUT_DIR = BASE / "dev_artifact_audit"
INVENTORY = OUTPUT_DIR / "artifact_inventory.tsv"
PASS_MARKER = OUTPUT_DIR / "DEV_ARTIFACT_AUDIT_PASS"


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def finite(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if not (
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.object_)
    ):
        return False
    return bool(np.all(np.isfinite(array)))


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if not SWEEP_MARKER.is_file():
        raise RuntimeError(f"Development sweep marker missing: {SWEEP_MARKER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PASS_MARKER.exists():
        PASS_MARKER.unlink()

    with MANIFEST.open(newline="", encoding="utf-8") as input_file:
        runs = list(csv.DictReader(input_file, delimiter="\t"))

    if len(runs) != 6:
        raise RuntimeError(f"Expected six training runs, found {len(runs)}")

    expected_run_keys = {
        (method, seed)
        for method in ("central", "dm2")
        for seed in (47, 48, 49)
    }
    actual_run_keys = {(row["method"], int(row["seed"])) for row in runs}
    if actual_run_keys != expected_run_keys:
        raise RuntimeError(
            f"Training-run manifest mismatch: expected={expected_run_keys}, "
            f"actual={actual_run_keys}"
        )

    inventory = []
    valid_sets = 0
    counters: Counter[str] = Counter()
    total_bytes = 0

    for row in runs:
        method = row["method"]
        seed = int(row["seed"])
        run_dir = ROOT / row["run_dir"]

        for iteration in ITERATIONS:
            eval_name = (
                f"part_b_dev_moderate_{method}_train{seed}_i{iteration}_"
                f"scenario{SCENARIO_SEED}_n{NUM_ENVS}"
            )
            directory = run_dir / "eval_logs" / eval_name
            problems = []

            if not directory.is_dir():
                problems.append(f"missing directory: {directory}")
            else:
                for filename, expected_shape in EXPECTED_SHAPES.items():
                    path = directory / filename
                    if not path.is_file() or path.stat().st_size == 0:
                        problems.append(f"missing/empty: {filename}")
                        continue

                    try:
                        array = np.load(path, mmap_mode="r", allow_pickle=False)
                    except Exception as error:
                        problems.append(f"{filename}: load failed: {error}")
                        continue

                    if tuple(array.shape) != expected_shape:
                        problems.append(
                            f"{filename}: shape={tuple(array.shape)}, "
                            f"expected={expected_shape}"
                        )
                    if filename == "death_status.npy" and array.dtype != np.bool_:
                        problems.append(
                            f"{filename}: dtype={array.dtype}, expected bool"
                        )
                    elif filename != "death_status.npy" and not np.issubdtype(
                        array.dtype, np.number
                    ):
                        problems.append(f"{filename}: nonnumeric dtype={array.dtype}")
                    if not finite(array):
                        problems.append(f"{filename}: non-finite or invalid dtype")

                    size = path.stat().st_size
                    total_bytes += size
                    inventory.append(
                        {
                            "method": method,
                            "train_seed": seed,
                            "iteration": iteration,
                            "scenario_seed": SCENARIO_SEED,
                            "evaluation": eval_name,
                            "artifact": filename,
                            "shape": repr(tuple(array.shape)),
                            "dtype": str(array.dtype),
                            "bytes": size,
                            "path": relative(path),
                        }
                    )
                    del array

            if problems:
                print(f"INVALID: {method} seed {seed} i{iteration}")
                for problem in problems:
                    print(f"  {problem}")
                raise RuntimeError("Development artifact validation failed")

            valid_sets += 1
            counters[method] += 1
            print(f"PASS: {method} seed {seed} i{iteration}")

    expected_sets = 2 * 3 * len(ITERATIONS)
    expected_rows = expected_sets * len(EXPECTED_SHAPES)
    if valid_sets != expected_sets or len(inventory) != expected_rows:
        raise RuntimeError(
            f"Inventory totals mismatch: sets={valid_sets}/{expected_sets}, "
            f"rows={len(inventory)}/{expected_rows}"
        )

    with INVENTORY.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(inventory[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(inventory)

    marker_text = (
        "Part B development artifact audit passed\n"
        f"Development scenario seed: {SCENARIO_SEED}\n"
        f"Evaluation environments: {NUM_ENVS}\n"
        f"Evaluations: {valid_sets}/{expected_sets}\n"
        f"Artifacts per evaluation: {len(EXPECTED_SHAPES)}\n"
        f"Inventory rows: {len(inventory)}\n"
        f"Artifact bytes: {total_bytes}\n"
    )
    PASS_MARKER.write_text(marker_text, encoding="utf-8")

    print("\n===== Development artifact audit =====")
    print(f"central: {counters['central']}/93")
    print(f"dm2:     {counters['dm2']}/93")
    print(f"PASS: {valid_sets}/{expected_sets} artifact sets are valid")
    print(f"Inventory: {relative(INVENTORY)}")
    print(f"Marker:    {relative(PASS_MARKER)}")


if __name__ == "__main__":
    main()
