#!/usr/bin/env python3
"""Validate the 18 locked Part B held-out evaluation artifact sets."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

import numpy as np


NUM_ENVS = 64
NUM_STEPS = 3000
SCENARIO_SEEDS = (107, 108)

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
LOCK_FILE = BASE / "heldout_protocol" / "locked_checkpoints.tsv"
LOCK_MARKER = BASE / "heldout_protocol" / "HELDOUT_PROTOCOL_LOCKED"
EVALUATION_MARKER = BASE / "heldout_protocol" / "HELDOUT_EVALUATIONS_PASS"
OUTPUT_DIR = BASE / "heldout_artifact_audit"
INVENTORY = OUTPUT_DIR / "artifact_inventory.tsv"
PASS_MARKER = OUTPUT_DIR / "HELDOUT_ARTIFACT_AUDIT_PASS"


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if not np.issubdtype(array.dtype, np.number):
        return False
    return bool(np.all(np.isfinite(array)))


def main() -> None:
    for marker in (LOCK_MARKER, EVALUATION_MARKER):
        if not marker.is_file():
            raise RuntimeError(f"Required marker missing: {marker}")
    if not LOCK_FILE.is_file():
        raise FileNotFoundError(LOCK_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PASS_MARKER.exists():
        PASS_MARKER.unlink()

    with LOCK_FILE.open(newline="", encoding="utf-8") as input_file:
        locked = list(csv.DictReader(input_file, delimiter="\t"))
    if len(locked) != 9:
        raise RuntimeError(f"Expected nine locked rows, found {len(locked)}")

    inventory = []
    valid_sets = 0
    total_bytes = 0

    for row in locked:
        role = row["role"]
        method = row["method"]
        seed = int(row["train_seed"])
        iteration = int(row["iteration"])
        checkpoint = ROOT / row["checkpoint"]

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_hash = sha256(checkpoint)
        if actual_hash != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")

        for scenario_seed in SCENARIO_SEEDS:
            eval_name = (
                f"part_b_final_heldout_{role}_{method}_train{seed}_i{iteration}_"
                f"scenario{scenario_seed}_n{NUM_ENVS}"
            )
            directory = checkpoint.parent / "eval_logs" / eval_name
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
                            "role": role,
                            "method": method,
                            "train_seed": seed,
                            "iteration": iteration,
                            "scenario_seed": scenario_seed,
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
                print(f"INVALID: {role} seed {seed} scenario {scenario_seed}")
                for problem in problems:
                    print(f"  {problem}")
                raise RuntimeError("Held-out artifact validation failed")

            valid_sets += 1
            print(f"PASS: {role} seed {seed} scenario {scenario_seed}")

    expected_sets = 18
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

    PASS_MARKER.write_text(
        "Part B held-out artifact audit passed\n"
        f"Held-out scenario seeds: {','.join(map(str, SCENARIO_SEEDS))}\n"
        f"Evaluation environments: {NUM_ENVS}\n"
        f"Evaluations: {valid_sets}/{expected_sets}\n"
        f"Artifacts per evaluation: {len(EXPECTED_SHAPES)}\n"
        f"Inventory rows: {len(inventory)}\n"
        f"Artifact bytes: {total_bytes}\n",
        encoding="utf-8",
    )

    print("\n===== Held-out artifact audit =====")
    print(f"PASS: {valid_sets}/{expected_sets} artifact sets are valid")
    print(f"Inventory: {relative(INVENTORY)}")
    print(f"Marker:    {relative(PASS_MARKER)}")


if __name__ == "__main__":
    main()
