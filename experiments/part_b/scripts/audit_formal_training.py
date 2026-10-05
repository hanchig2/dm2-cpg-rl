#!/usr/bin/env python3
"""Audit the six Part B training runs and build the development manifest."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml


SOURCE_COMMIT = "9ae10d2d8e2c5760b41439f4b9d42a190ee2eacf"
TRAINING_SEEDS = (47, 48, 49)
ITERATIONS = tuple(range(0, 3000, 100)) + (2999,)

METHODS = {
    "central": {
        "experiment": "part_b_central_from_scratch_static_mixed_cpg_unitree_a1",
    },
    "dm2": {
        "experiment": "part_b_dm2_message_from_scratch_static_mixed_cpg_unitree_a1",
    },
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
OUTPUT_DIR = ROOT / "analysis_outputs" / "part_b_from_scratch" / "formal_audit"
MANIFEST = OUTPUT_DIR / "formal_run_manifest.tsv"
MARKER = OUTPUT_DIR / "FORMAL_TRAINING_AUDIT_PASS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as input_file:
        return yaml.load(input_file, Loader=yaml.UnsafeLoader)


def member(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def tensors(value: Any, prefix: str = "checkpoint") -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensors(child, f"{prefix}[{index}]")


def checkpoint_layout(checkpoint: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("model_state_dict is missing or empty")
    return tuple(
        sorted(
            (name, tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in state.items()
            if torch.is_tensor(tensor)
        )
    )


def checkpoint_iteration(checkpoint: dict[str, Any]) -> int | None:
    value = checkpoint.get("iter")
    if value is None:
        value = checkpoint.get("iteration")
    if value is None:
        return None
    return int(value)


def assert_source_commit() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )


def discover_run(method: str, seed: int) -> Path:
    experiment = str(METHODS[method]["experiment"])
    experiment_dir = ROOT / "logs" / "rsl_rl" / experiment
    run_name = f"part_b_{method}_from_scratch_train{seed}_i3000"

    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory missing: {experiment_dir}")

    matches = sorted(
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and path.name.endswith(f"_{run_name}")
    )
    complete = [path for path in matches if (path / "model_2999.pt").is_file()]

    if len(complete) != 1:
        raise RuntimeError(
            f"Expected one complete run for {method} seed {seed}; "
            f"found {len(complete)} among {len(matches)} matching directories"
        )
    return complete[0]


def validate_console(method: str, seed: int) -> None:
    run_name = f"part_b_{method}_from_scratch_train{seed}_i3000"
    console = ROOT / "logs" / "rsl_rl" / f"{run_name}.console.log"
    if not console.is_file():
        raise FileNotFoundError(f"Console log missing: {console}")
    text = console.read_text(encoding="utf-8", errors="replace")
    if "Loading model checkpoint from" in text:
        raise RuntimeError(f"Run was not from scratch: {console}")
    if not re.search(r"Learning iteration\s+2999/3000", text):
        raise RuntimeError(f"Final learning iteration missing from {console}")


def validate_config(method: str, run_dir: Path) -> None:
    env_path = run_dir / "params" / "env.yaml"
    agent_path = run_dir / "params" / "agent.yaml"
    if not env_path.is_file() or not agent_path.is_file():
        raise FileNotFoundError(f"Saved parameter files missing in {run_dir}")

    env_cfg = load_yaml(env_path)
    agent_cfg = load_yaml(agent_path)

    scene = member(env_cfg, "scene")
    saved_num_envs = member(scene, "num_envs")
    if saved_num_envs is not None and int(saved_num_envs) != 64:
        raise RuntimeError(f"{run_dir}: num_envs={saved_num_envs}, expected 64")

    saved_iterations = member(agent_cfg, "max_iterations")
    if saved_iterations is None or int(saved_iterations) != 3000:
        raise RuntimeError(
            f"{run_dir}: max_iterations={saved_iterations}, expected 3000"
        )

    expected_experiment = METHODS[method]["experiment"]
    saved_experiment = member(agent_cfg, "experiment_name")
    if saved_experiment != expected_experiment:
        raise RuntimeError(
            f"{run_dir}: experiment_name={saved_experiment!r}, "
            f"expected {expected_experiment!r}"
        )


def validate_checkpoints(method: str, seed: int, run_dir: Path) -> tuple[str, int]:
    expected_names = {f"model_{iteration}.pt" for iteration in ITERATIONS}
    actual = {
        path.name: path
        for path in run_dir.glob("model_*.pt")
        if re.fullmatch(r"model_[0-9]+\.pt", path.name)
    }
    actual_names = set(actual)
    if actual_names != expected_names:
        raise RuntimeError(
            f"{run_dir}: checkpoint mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    reference_layout = None
    for iteration in ITERATIONS:
        path = actual[f"model_{iteration}.pt"]
        if path.stat().st_size == 0:
            raise RuntimeError(f"Empty checkpoint: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Checkpoint is not a dictionary: {path}")

        stored_iteration = checkpoint_iteration(checkpoint)
        if stored_iteration is not None and stored_iteration != iteration:
            raise RuntimeError(
                f"{path}: stored iteration {stored_iteration}, expected {iteration}"
            )

        layout = checkpoint_layout(checkpoint)
        if reference_layout is None:
            reference_layout = layout
        elif layout != reference_layout:
            raise RuntimeError(f"Model tensor layout changed within run: {path}")

        bad_tensors = []
        for name, tensor in tensors(checkpoint.get("model_state_dict", {}), "model"):
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                bad_tensors.append(name)
        if bad_tensors:
            raise RuntimeError(f"Non-finite model tensors in {path}: {bad_tensors}")

        del checkpoint

    endpoint = run_dir / "model_2999.pt"
    print(f"PASS: {method} seed {seed}: {len(actual)} checkpoints")
    return sha256(endpoint), len(actual)


def main() -> None:
    assert_source_commit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MARKER.exists():
        MARKER.unlink()

    rows: list[dict[str, str | int]] = []
    model_layouts: dict[str, tuple[tuple[str, tuple[int, ...], str], ...]] = {}

    for seed in TRAINING_SEEDS:
        for method in METHODS:
            run_dir = discover_run(method, seed)
            validate_console(method, seed)
            validate_config(method, run_dir)
            endpoint_hash, count = validate_checkpoints(method, seed, run_dir)

            endpoint_checkpoint = torch.load(
                run_dir / "model_2999.pt", map_location="cpu", weights_only=False
            )
            layout = checkpoint_layout(endpoint_checkpoint)
            if method in model_layouts and model_layouts[method] != layout:
                raise RuntimeError(f"{method}: endpoint layouts differ across seeds")
            model_layouts[method] = layout
            del endpoint_checkpoint

            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "run_dir": relative(run_dir),
                    "checkpoint": relative(run_dir / "model_2999.pt"),
                    "checkpoint_sha256": endpoint_hash,
                    "checkpoint_count": count,
                }
            )

    if len(rows) != 6:
        raise RuntimeError(f"Expected six audited runs, found {len(rows)}")

    with MANIFEST.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "method",
                "seed",
                "run_dir",
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_count",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    marker_text = (
        "Part B formal training audit passed\n"
        "Methods: central,dm2\n"
        "Training seeds: 47,48,49\n"
        "Runs: 6/6\n"
        "Checkpoints per run: 31\n"
        "Final iteration: 2999\n"
        "Training mode: from scratch\n"
        f"Source commit: {SOURCE_COMMIT}\n"
    )
    MARKER.write_text(marker_text, encoding="utf-8")

    print("\n===== Formal training audit =====")
    print("PASS: all six Part B runs are complete and structurally valid")
    print(f"Manifest: {relative(MANIFEST)}")
    print(f"Marker:   {relative(MARKER)}")


if __name__ == "__main__":
    main()
