#!/usr/bin/env python3
"""Lock aggregate development-selected checkpoints before held-out evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_COMMIT = "9ae10d2d8e2c5760b41439f4b9d42a190ee2eacf"
DEVELOPMENT_SEED = 101
HELDOUT_SEEDS = (107, 108)
NUM_ENVS = 64
STEPS_PER_UPDATE = 64 * 48


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
ANALYSIS_MARKER = BASE / "dev_learning_curves" / "DEV_ANALYSIS_PASS"
RANKING = BASE / "dev_learning_curves" / "dev_checkpoint_ranking.csv"
METHOD_CURVE = BASE / "dev_learning_curves" / "dev_method_curve_corrected.csv"
RESULT_JSON = BASE / "dev_learning_curves" / "sample_efficiency_results.json"
OUTPUT_DIR = BASE / "heldout_protocol"
LOCK_FILE = OUTPUT_DIR / "locked_checkpoints.tsv"
PROTOCOL_JSON = OUTPUT_DIR / "heldout_protocol.json"
LOCK_MARKER = OUTPUT_DIR / "HELDOUT_PROTOCOL_LOCKED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file, delimiter=delimiter))


def assert_no_heldout_directories() -> None:
    eval_roots = ROOT.glob("logs/rsl_rl/*/*/eval_logs")
    found = []
    for eval_root in eval_roots:
        for seed in HELDOUT_SEEDS:
            found.extend(eval_root.glob(f"*scenario{seed}_n*"))
    if found:
        raise RuntimeError(
            "Held-out evaluation directories already exist before locking: "
            + ", ".join(relative(path) for path in found)
        )


def ranking_row(rows: list[dict[str, str]], method: str, iteration: int) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == method and int(row["iteration"]) == iteration
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one ranking row for {method} i{iteration}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )

    for path in (MANIFEST, RANKING, METHOD_CURVE, RESULT_JSON):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not ANALYSIS_MARKER.is_file():
        raise RuntimeError(f"Development analysis marker missing: {ANALYSIS_MARKER}")

    assert_no_heldout_directories()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_MARKER.exists():
        LOCK_MARKER.unlink()

    runs = read_csv(MANIFEST, delimiter="\t")
    ranking = read_csv(RANKING)
    results = json.loads(RESULT_JSON.read_text(encoding="utf-8"))

    central_iteration = int(
        results["methods"]["central"]["development_best_checkpoint"]["iteration"]
    )
    dm2_iteration = int(
        results["methods"]["dm2"]["development_best_checkpoint"]["iteration"]
    )
    matched_iteration = central_iteration

    selections = (
        (
            "central_selected_and_matched",
            "central",
            central_iteration,
            "Aggregate development optimum and matched-budget anchor.",
        ),
        (
            "dm2_matched_budget",
            "dm2",
            matched_iteration,
            "Strictly matched to the Central development-selected interaction budget.",
        ),
        (
            "dm2_development_selected",
            "dm2",
            dm2_iteration,
            "Highest mean development LV across the three training seeds.",
        ),
    )

    locked = []
    for role, method, iteration, reason in selections:
        aggregate = ranking_row(ranking, method, iteration)
        method_runs = sorted(
            (row for row in runs if row["method"] == method),
            key=lambda row: int(row["seed"]),
        )
        if len(method_runs) != 3:
            raise RuntimeError(f"Expected three {method} runs, found {len(method_runs)}")

        interactions = (iteration + 1) * STEPS_PER_UPDATE
        for run in method_runs:
            checkpoint = ROOT / run["run_dir"] / f"model_{iteration}.pt"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(checkpoint)
            locked.append(
                {
                    "role": role,
                    "method": method,
                    "train_seed": int(run["seed"]),
                    "iteration": iteration,
                    "interactions": interactions,
                    "dev_lv_mean": float(aggregate["lv_rate_mean"]),
                    "dev_lv_std": float(aggregate["lv_rate_std"]),
                    "checkpoint": relative(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "selection_reason": reason,
                }
            )

    if len(locked) != 9:
        raise RuntimeError(f"Expected nine locked checkpoints, found {len(locked)}")

    with LOCK_FILE.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(locked[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(locked)

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "development_scenario_seed": DEVELOPMENT_SEED,
        "heldout_scenario_seeds": list(HELDOUT_SEEDS),
        "num_environments_per_evaluation": NUM_ENVS,
        "selection_unit": "One common iteration per method across all training seeds.",
        "primary_comparison": {
            "central_iteration": central_iteration,
            "dm2_iteration": matched_iteration,
            "interactions_per_run": (central_iteration + 1) * STEPS_PER_UPDATE,
            "description": "Strict matched-interaction comparison.",
        },
        "secondary_comparison": {
            "central_iteration": central_iteration,
            "dm2_iteration": dm2_iteration,
            "central_interactions": (central_iteration + 1) * STEPS_PER_UPDATE,
            "dm2_interactions": (dm2_iteration + 1) * STEPS_PER_UPDATE,
            "description": "Method-specific development-selected comparison.",
        },
        "locked_checkpoint_count": len(locked),
        "planned_evaluation_count": len(locked) * len(HELDOUT_SEEDS),
        "locked_checkpoints_tsv": relative(LOCK_FILE),
        "development_curve_sha256": sha256(METHOD_CURVE),
        "selection_policy": (
            "Aggregate development results select one checkpoint per method; "
            "DM2 is also evaluated at the Central interaction budget; no per-seed "
            "checkpoint selection is permitted."
        ),
    }
    PROTOCOL_JSON.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    LOCK_MARKER.write_text(
        "Part B held-out protocol locked\n"
        f"Development scenario seed: {DEVELOPMENT_SEED}\n"
        f"Held-out scenario seeds: {','.join(map(str, HELDOUT_SEEDS))}\n"
        f"Central selected iteration: {central_iteration}\n"
        f"DM2 matched iteration: {matched_iteration}\n"
        f"DM2 selected iteration: {dm2_iteration}\n"
        f"Locked checkpoints: {len(locked)}\n"
        f"Planned evaluations: {len(locked) * len(HELDOUT_SEEDS)}\n",
        encoding="utf-8",
    )

    print("===== Held-out protocol lock =====")
    print(f"Central selected: i{central_iteration}")
    print(f"DM2 matched:     i{matched_iteration}")
    print(f"DM2 selected:    i{dm2_iteration}")
    print(f"PASS: {len(locked)} checkpoints locked before held-out evaluation")
    print(f"Protocol: {relative(PROTOCOL_JSON)}")
    print(f"Lock:     {relative(LOCK_FILE)}")


if __name__ == "__main__":
    main()
