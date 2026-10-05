import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DT = 0.01
GRAVITY = 9.80665
MAX_STEPS = 3000
MAX_TIME = MAX_STEPS * DT

NUM_ENVS = 64
SCENARIO_SEED = 101
STEPS_PER_UPDATE = 64 * 48

MIN_PROGRESS_RATIO = 0.70
MIN_STRAIGHTNESS = 0.80
MAX_ABS_YAW_BIAS = 0.05
MAX_TRANSLATION_RMSE = 0.30
TARGET_LV_RATE = 0.95

ITERATIONS = tuple(range(0, 3000, 100)) + (2999,)

MANIFEST = Path(
    "analysis_outputs/part_b_from_scratch/"
    "formal_audit/formal_run_manifest.tsv"
)

AUDIT_MARKER = Path(
    "analysis_outputs/part_b_from_scratch/"
    "dev_artifact_audit/"
    "DEV_ARTIFACT_AUDIT_PASS"
)

OUTPUT_DIR = Path(
    "analysis_outputs/part_b_from_scratch/"
    "dev_learning_curves"
)

ROLLOUT_CSV = OUTPUT_DIR / "dev_rollouts_corrected.csv"
SEED_SUMMARY_CSV = OUTPUT_DIR / "dev_seed_summary_corrected.csv"
METHOD_CURVE_CSV = OUTPUT_DIR / "dev_method_curve_corrected.csv"
RANKING_CSV = OUTPUT_DIR / "dev_checkpoint_ranking.csv"
RESULT_JSON = OUTPUT_DIR / "sample_efficiency_results.json"

LV_FIGURE = OUTPUT_DIR / "lv_learning_curve.png"
LV_FIGURE_PDF = OUTPUT_DIR / "lv_learning_curve.pdf"
CRITERIA_FIGURE = OUTPUT_DIR / "criterion_learning_curves.png"
PASS_MARKER = OUTPUT_DIR / "DEV_ANALYSIS_PASS"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if PASS_MARKER.exists():
    PASS_MARKER.unlink()


def load(directory, filename):
    path = directory / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    value = np.asarray(
        np.load(
            path,
            allow_pickle=False,
        )
    )

    if (
        np.issubdtype(value.dtype, np.number)
        and not np.all(np.isfinite(value))
    ):
        raise RuntimeError(
            f"Non-finite values in {path}"
        )

    return value


def quaternion_yaw_wxyz(quaternion):
    w, x, y, z = [
        float(component)
        for component in quaternion
    ]

    norm = math.sqrt(
        w * w
        + x * x
        + y * y
        + z * z
    )

    if norm <= 1.0e-12:
        raise RuntimeError(
            "Zero-norm initial quaternion."
        )

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def mean(rows, field):
    values = np.asarray(
        [
            float(row[field])
            for row in rows
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(values)

    if not np.any(finite):
        return float("nan")

    return float(np.mean(values[finite]))


def sample_std(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size <= 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows available for {path}"
        )

    with path.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


if not MANIFEST.is_file():
    raise FileNotFoundError(MANIFEST)

if not AUDIT_MARKER.is_file():
    raise RuntimeError(
        "Development-artifact audit marker missing."
    )

with MANIFEST.open(newline="") as input_file:
    runs = list(
        csv.DictReader(
            input_file,
            delimiter="\t",
        )
    )

if len(runs) != 6:
    raise RuntimeError(
        f"Expected six runs, got {len(runs)}."
    )


rollout_rows = []
maximum_survival_discrepancy = 0.0


for run in runs:
    method = run["method"]
    train_seed = int(run["seed"])
    run_dir = Path(run["run_dir"])

    for iteration in ITERATIONS:
        eval_name = (
            f"part_b_dev_moderate_"
            f"{method}_train{train_seed}_"
            f"i{iteration}_"
            f"scenario{SCENARIO_SEED}_"
            f"n{NUM_ENVS}"
        )

        directory = (
            run_dir
            / "eval_logs"
            / eval_name
        )

        print(
            f"Analyzing {method} "
            f"seed {train_seed} "
            f"model_{iteration}"
        )

        angular_velocity = load(
            directory,
            "base_angular_velocity.npy",
        ).astype(np.float64)

        position = load(
            directory,
            "base_position.npy",
        ).astype(np.float64)

        quaternion = load(
            directory,
            "base_quaternion.npy",
        ).astype(np.float64)

        tracked_velocity = load(
            directory,
            "base_tracked_velocity.npy",
        ).astype(np.float64)

        command = load(
            directory,
            "command.npy",
        ).astype(np.float64)

        energy = load(
            directory,
            "energy.npy",
        ).astype(np.float64)

        survival_mask = load(
            directory,
            "survival_mask.npy",
        ).astype(np.float64)

        raw_survival_time = load(
            directory,
            "survival_time.npy",
        ).astype(np.float64)

        mass_value = load(
            directory,
            "robot_mass.npy",
        )

        expected_shapes = {
            "angular_velocity": (
                angular_velocity,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            "position": (
                position,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            "quaternion": (
                quaternion,
                (NUM_ENVS, 4, MAX_STEPS),
            ),
            "tracked_velocity": (
                tracked_velocity,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            "command": (
                command,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            "energy": (
                energy,
                (NUM_ENVS, 1, MAX_STEPS),
            ),
            "survival_mask": (
                survival_mask,
                (NUM_ENVS, 1, MAX_STEPS),
            ),
            "survival_time": (
                raw_survival_time,
                (NUM_ENVS,),
            ),
        }

        for name, (
            value,
            expected_shape,
        ) in expected_shapes.items():
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"{directory}: {name} "
                    f"shape {value.shape}, "
                    f"expected {expected_shape}"
                )

        robot_mass = float(
            np.asarray(mass_value).reshape(())
        )

        if (
            not math.isfinite(robot_mass)
            or robot_mass <= 0.0
        ):
            raise RuntimeError(
                f"Invalid robot mass: {robot_mass}"
            )

        mask = (
            survival_mask[:, 0, :]
            > 0.5
        )

        mask_survival_time = (
            np.sum(
                mask,
                axis=1,
            )
            * DT
        )

        if np.max(raw_survival_time) > (
            MAX_TIME + 1.0
        ):
            saved_survival_time = (
                raw_survival_time
                * DT
            )
        else:
            saved_survival_time = (
                raw_survival_time.copy()
            )

        saved_survival_time = np.clip(
            saved_survival_time,
            0.0,
            MAX_TIME,
        )

        discrepancy = float(
            np.max(
                np.abs(
                    saved_survival_time
                    - mask_survival_time
                )
            )
        )

        maximum_survival_discrepancy = max(
            maximum_survival_discrepancy,
            discrepancy,
        )

        if discrepancy > 2.1 * DT:
            raise RuntimeError(
                f"Survival-time mismatch in "
                f"{directory}: {discrepancy}"
            )

        survival_time = mask_survival_time

        for environment_index in range(
            NUM_ENVS
        ):
            valid_indices = np.flatnonzero(
                mask[environment_index]
            )

            if valid_indices.size == 0:
                raise RuntimeError(
                    f"No valid steps for "
                    f"{directory}, env "
                    f"{environment_index}"
                )

            first_index = int(
                valid_indices[0]
            )

            last_index = int(
                valid_indices[-1]
            )

            valid = valid_indices

            actual_xy = (
                tracked_velocity[
                    environment_index,
                    0:2,
                    :,
                ][
                    :,
                    valid,
                ].T
            )

            target_xy = (
                command[
                    environment_index,
                    0:2,
                    :,
                ][
                    :,
                    valid,
                ].T
            )

            vector_error = (
                actual_xy
                - target_xy
            )

            translation_rmse = math.sqrt(
                float(
                    np.mean(
                        np.sum(
                            vector_error
                            * vector_error,
                            axis=1,
                        )
                    )
                )
            )

            yaw_error = (
                angular_velocity[
                    environment_index,
                    2,
                    valid,
                ]
                - command[
                    environment_index,
                    2,
                    valid,
                ]
            )

            abs_yaw_bias = abs(
                float(
                    np.mean(yaw_error)
                )
            )

            initial_yaw = (
                quaternion_yaw_wxyz(
                    quaternion[
                        environment_index,
                        :,
                        first_index,
                    ]
                )
            )

            initial_heading = np.asarray(
                [
                    math.cos(initial_yaw),
                    math.sin(initial_yaw),
                ],
                dtype=np.float64,
            )

            trajectory = (
                position[
                    environment_index,
                    0:2,
                    :,
                ][
                    :,
                    valid,
                ].T
            )

            displacement = (
                trajectory[-1]
                - trajectory[0]
            )

            progress = float(
                np.dot(
                    displacement,
                    initial_heading,
                )
            )

            if trajectory.shape[0] > 1:
                path_segments = np.diff(
                    trajectory,
                    axis=0,
                )

                path_length = float(
                    np.sum(
                        np.linalg.norm(
                            path_segments,
                            axis=1,
                        )
                    )
                )
            else:
                path_length = 0.0

            desired_speed = np.linalg.norm(
                target_xy,
                axis=1,
            )

            commanded_distance = float(
                np.sum(desired_speed)
                * DT
            )

            if commanded_distance > 1.0e-12:
                progress_ratio = (
                    progress
                    / commanded_distance
                )
            else:
                progress_ratio = 0.0

            if path_length > 1.0e-12:
                straightness = (
                    max(progress, 0.0)
                    / path_length
                )
            else:
                straightness = 0.0

            straightness = float(
                np.clip(
                    straightness,
                    0.0,
                    1.0,
                )
            )

            total_energy = float(
                np.sum(
                    energy[
                        environment_index,
                        0,
                        valid,
                    ]
                )
                * DT
            )

            if path_length > 1.0e-12:
                path_cot = (
                    total_energy
                    / (
                        robot_mass
                        * GRAVITY
                        * path_length
                    )
                )
            else:
                path_cot = float("nan")

            survived = int(
                survival_time[
                    environment_index
                ]
                >= MAX_TIME - 0.5 * DT
            )

            progress_pass = int(
                progress_ratio
                >= MIN_PROGRESS_RATIO
            )

            straightness_pass = int(
                straightness
                >= MIN_STRAIGHTNESS
            )

            yaw_pass = int(
                abs_yaw_bias
                <= MAX_ABS_YAW_BIAS
            )

            translation_pass = int(
                translation_rmse
                <= MAX_TRANSLATION_RMSE
            )

            balance_pass = int(
                progress_pass
                and straightness_pass
                and yaw_pass
                and translation_pass
            )

            lv_pass = int(
                survived
                and balance_pass
            )

            rollout_rows.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "scenario_seed": (
                        SCENARIO_SEED
                    ),
                    "iteration": iteration,
                    "interactions": (
                        (iteration + 1)
                        * STEPS_PER_UPDATE
                    ),
                    "environment_index": (
                        environment_index
                    ),
                    "survival_time": float(
                        survival_time[
                            environment_index
                        ]
                    ),
                    "survived": survived,
                    "translation_rmse": (
                        translation_rmse
                    ),
                    "abs_yaw_bias": (
                        abs_yaw_bias
                    ),
                    "progress": progress,
                    "commanded_distance": (
                        commanded_distance
                    ),
                    "progress_ratio": (
                        progress_ratio
                    ),
                    "path_length": (
                        path_length
                    ),
                    "straightness": (
                        straightness
                    ),
                    "total_energy": (
                        total_energy
                    ),
                    "robot_mass": robot_mass,
                    "path_cot": path_cot,
                    "progress_pass": (
                        progress_pass
                    ),
                    "straightness_pass": (
                        straightness_pass
                    ),
                    "yaw_pass": yaw_pass,
                    "translation_pass": (
                        translation_pass
                    ),
                    "balance_pass": (
                        balance_pass
                    ),
                    "lv_pass": lv_pass,
                }
            )

        del (
            angular_velocity,
            position,
            quaternion,
            tracked_velocity,
            command,
            energy,
            survival_mask,
            raw_survival_time,
            mask,
        )

        gc.collect()


expected_rollouts = (
    2
    * 3
    * len(ITERATIONS)
    * NUM_ENVS
)

if len(rollout_rows) != expected_rollouts:
    raise RuntimeError(
        f"Expected {expected_rollouts} rollouts, "
        f"got {len(rollout_rows)}."
    )


groups = defaultdict(list)

for row in rollout_rows:
    groups[
        (
            row["method"],
            row["train_seed"],
            row["iteration"],
        )
    ].append(row)


seed_summary_rows = []

for (
    method,
    train_seed,
    iteration,
), rows in sorted(groups.items()):
    if len(rows) != NUM_ENVS:
        raise RuntimeError(
            f"Expected {NUM_ENVS} rollouts for "
            f"{method} seed {train_seed} "
            f"iteration {iteration}, "
            f"got {len(rows)}."
        )

    seed_summary_rows.append(
        {
            "method": method,
            "train_seed": train_seed,
            "scenario_seed": SCENARIO_SEED,
            "iteration": iteration,
            "interactions": (
                (iteration + 1)
                * STEPS_PER_UPDATE
            ),
            "num_rollouts": len(rows),
            "survival_rate": mean(
                rows,
                "survived",
            ),
            "rmst": mean(
                rows,
                "survival_time",
            ),
            "translation_rmse": mean(
                rows,
                "translation_rmse",
            ),
            "abs_yaw_bias": mean(
                rows,
                "abs_yaw_bias",
            ),
            "progress": mean(
                rows,
                "progress",
            ),
            "progress_ratio": mean(
                rows,
                "progress_ratio",
            ),
            "straightness": mean(
                rows,
                "straightness",
            ),
            "path_cot": mean(
                rows,
                "path_cot",
            ),
            "progress_pass_rate": mean(
                rows,
                "progress_pass",
            ),
            "straightness_pass_rate": mean(
                rows,
                "straightness_pass",
            ),
            "yaw_pass_rate": mean(
                rows,
                "yaw_pass",
            ),
            "translation_pass_rate": mean(
                rows,
                "translation_pass",
            ),
            "balance_pass_rate": mean(
                rows,
                "balance_pass",
            ),
            "lv_rate": mean(
                rows,
                "lv_pass",
            ),
        }
    )


curve_groups = defaultdict(list)

for row in seed_summary_rows:
    curve_groups[
        (
            row["method"],
            row["iteration"],
        )
    ].append(row)


curve_metrics = (
    "survival_rate",
    "rmst",
    "translation_rmse",
    "abs_yaw_bias",
    "progress",
    "progress_ratio",
    "straightness",
    "path_cot",
    "progress_pass_rate",
    "straightness_pass_rate",
    "yaw_pass_rate",
    "translation_pass_rate",
    "balance_pass_rate",
    "lv_rate",
)

method_curve_rows = []

for (
    method,
    iteration,
), rows in sorted(curve_groups.items()):
    if len(rows) != 3:
        raise RuntimeError(
            f"Expected three training seeds for "
            f"{method} iteration {iteration}, "
            f"got {len(rows)}."
        )

    record = {
        "method": method,
        "iteration": iteration,
        "interactions": (
            (iteration + 1)
            * STEPS_PER_UPDATE
        ),
        "num_training_seeds": 3,
        "num_rollouts": (
            3 * NUM_ENVS
        ),
    }

    for metric in curve_metrics:
        values = [
            float(row[metric])
            for row in rows
        ]

        record[f"{metric}_mean"] = float(
            np.mean(values)
        )

        record[f"{metric}_std"] = (
            sample_std(values)
        )

    method_curve_rows.append(record)


ranking_rows = []

for method in ("central", "dm2"):
    method_rows = [
        row
        for row in method_curve_rows
        if row["method"] == method
    ]

    method_rows.sort(
        key=lambda row: (
            -row["lv_rate_mean"],
            -row["survival_rate_mean"],
            -row[
                "progress_pass_rate_mean"
            ],
            -row[
                "straightness_pass_rate_mean"
            ],
            row[
                "translation_rmse_mean"
            ],
            row["path_cot_mean"],
            row["iteration"],
        )
    )

    for rank, row in enumerate(
        method_rows,
        start=1,
    ):
        ranking_rows.append(
            {
                "method": method,
                "rank": rank,
                "iteration": row["iteration"],
                "interactions": (
                    row["interactions"]
                ),
                "lv_rate_mean": (
                    row["lv_rate_mean"]
                ),
                "lv_rate_std": (
                    row["lv_rate_std"]
                ),
                "survival_rate_mean": (
                    row["survival_rate_mean"]
                ),
                "progress_pass_rate_mean": (
                    row[
                        "progress_pass_rate_mean"
                    ]
                ),
                "straightness_pass_rate_mean": (
                    row[
                        "straightness_pass_rate_mean"
                    ]
                ),
                "yaw_pass_rate_mean": (
                    row["yaw_pass_rate_mean"]
                ),
                "translation_pass_rate_mean": (
                    row[
                        "translation_pass_rate_mean"
                    ]
                ),
                "translation_rmse_mean": (
                    row[
                        "translation_rmse_mean"
                    ]
                ),
                "progress_ratio_mean": (
                    row["progress_ratio_mean"]
                ),
                "straightness_mean": (
                    row["straightness_mean"]
                ),
                "path_cot_mean": (
                    row["path_cot_mean"]
                ),
            }
        )


results = {
    "scenario_seed": SCENARIO_SEED,
    "num_training_seeds": 3,
    "num_environments_per_seed": NUM_ENVS,
    "target_lv_rate": TARGET_LV_RATE,
    "maximum_survival_discrepancy_seconds": (
        maximum_survival_discrepancy
    ),
    "methods": {},
}


for method in ("central", "dm2"):
    rows = sorted(
        [
            row
            for row in method_curve_rows
            if row["method"] == method
        ],
        key=lambda row: row["iteration"],
    )

    threshold_rows = [
        row
        for row in rows
        if row["lv_rate_mean"]
        >= TARGET_LV_RATE
    ]

    first_threshold = (
        threshold_rows[0]
        if threshold_rows
        else None
    )

    best_row = next(
        row
        for row in ranking_rows
        if (
            row["method"] == method
            and row["rank"] == 1
        )
    )

    final_row = next(
        row
        for row in rows
        if row["iteration"] == 2999
    )

    interactions = np.asarray(
        [
            row["interactions"]
            for row in rows
        ],
        dtype=np.float64,
    )

    lv_values = np.asarray(
        [
            row["lv_rate_mean"]
            for row in rows
        ],
        dtype=np.float64,
    )

    normalized_auc = float(
        np.trapz(
            lv_values,
            interactions,
        )
        / (
            interactions[-1]
            - interactions[0]
        )
    )

    per_seed_threshold = {}

    for seed in (47, 48, 49):
        seed_rows = sorted(
            [
                row
                for row in seed_summary_rows
                if (
                    row["method"] == method
                    and row["train_seed"]
                    == seed
                )
            ],
            key=lambda row: row["iteration"],
        )

        crossing = next(
            (
                row
                for row in seed_rows
                if row["lv_rate"]
                >= TARGET_LV_RATE
            ),
            None,
        )

        per_seed_threshold[
            str(seed)
        ] = (
            None
            if crossing is None
            else {
                "iteration": (
                    crossing["iteration"]
                ),
                "interactions": (
                    crossing["interactions"]
                ),
                "lv_rate": (
                    crossing["lv_rate"]
                ),
            }
        )

    results["methods"][method] = {
        "first_95_percent_lv": (
            None
            if first_threshold is None
            else {
                "iteration": (
                    first_threshold["iteration"]
                ),
                "interactions": (
                    first_threshold[
                        "interactions"
                    ]
                ),
                "lv_rate_mean": (
                    first_threshold[
                        "lv_rate_mean"
                    ]
                ),
                "lv_rate_std": (
                    first_threshold[
                        "lv_rate_std"
                    ]
                ),
            }
        ),
        "per_training_seed_first_95_percent": (
            per_seed_threshold
        ),
        "normalized_lv_auc": normalized_auc,
        "development_best_checkpoint": {
            "iteration": best_row["iteration"],
            "interactions": (
                best_row["interactions"]
            ),
            "lv_rate_mean": (
                best_row["lv_rate_mean"]
            ),
            "lv_rate_std": (
                best_row["lv_rate_std"]
            ),
        },
        "final_checkpoint": {
            "iteration": 2999,
            "interactions": (
                final_row["interactions"]
            ),
            "lv_rate_mean": (
                final_row["lv_rate_mean"]
            ),
            "lv_rate_std": (
                final_row["lv_rate_std"]
            ),
        },
    }


write_csv(
    ROLLOUT_CSV,
    rollout_rows,
)

write_csv(
    SEED_SUMMARY_CSV,
    seed_summary_rows,
)

write_csv(
    METHOD_CURVE_CSV,
    method_curve_rows,
)

write_csv(
    RANKING_CSV,
    ranking_rows,
)

with RESULT_JSON.open("w") as output_file:
    json.dump(
        results,
        output_file,
        indent=2,
        sort_keys=True,
    )


colors = {
    "central": "#1f77b4",
    "dm2": "#d62728",
}

labels = {
    "central": "Centralized CPG-RL",
    "dm2": "DM²-CPG-RL",
}


fig, axis = plt.subplots(
    figsize=(9.0, 5.5),
)

for method in ("central", "dm2"):
    rows = sorted(
        [
            row
            for row in method_curve_rows
            if row["method"] == method
        ],
        key=lambda row: row["iteration"],
    )

    x = np.asarray(
        [
            row["interactions"]
            for row in rows
        ],
        dtype=np.float64,
    ) / 1.0e6

    y = np.asarray(
        [
            row["lv_rate_mean"]
            for row in rows
        ],
        dtype=np.float64,
    )

    error = np.asarray(
        [
            row["lv_rate_std"]
            for row in rows
        ],
        dtype=np.float64,
    )

    axis.plot(
        x,
        y,
        color=colors[method],
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label=labels[method],
    )

    axis.fill_between(
        x,
        np.clip(y - error, 0.0, 1.0),
        np.clip(y + error, 0.0, 1.0),
        color=colors[method],
        alpha=0.18,
    )

axis.axhline(
    TARGET_LV_RATE,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="95% LV target",
)

axis.set_xlabel(
    "Environment interactions (millions)"
)

axis.set_ylabel(
    "Locomotion Viability rate"
)

axis.set_ylim(
    0.0,
    1.02,
)

axis.grid(
    True,
    alpha=0.25,
)

axis.legend()
fig.tight_layout()

fig.savefig(
    LV_FIGURE,
    dpi=220,
)

fig.savefig(
    LV_FIGURE_PDF,
)

plt.close(fig)


criterion_panels = (
    (
        "survival_rate",
        "Survival rate",
        None,
    ),
    (
        "progress_pass_rate",
        "Progress criterion pass rate",
        MIN_PROGRESS_RATIO,
    ),
    (
        "straightness_pass_rate",
        "Straightness criterion pass rate",
        MIN_STRAIGHTNESS,
    ),
    (
        "translation_pass_rate",
        "Translation criterion pass rate",
        None,
    ),
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.0, 7.5),
    sharex=True,
    sharey=True,
)

for axis, (
    metric,
    title,
    threshold,
) in zip(
    axes.ravel(),
    criterion_panels,
):
    for method in ("central", "dm2"):
        rows = sorted(
            [
                row
                for row in method_curve_rows
                if row["method"] == method
            ],
            key=lambda row: row["iteration"],
        )

        x = np.asarray(
            [
                row["interactions"]
                for row in rows
            ],
            dtype=np.float64,
        ) / 1.0e6

        y = np.asarray(
            [
                row[f"{metric}_mean"]
                for row in rows
            ],
            dtype=np.float64,
        )

        error = np.asarray(
            [
                row[f"{metric}_std"]
                for row in rows
            ],
            dtype=np.float64,
        )

        axis.plot(
            x,
            y,
            color=colors[method],
            linewidth=1.8,
            label=labels[method],
        )

        axis.fill_between(
            x,
            np.clip(y - error, 0.0, 1.0),
            np.clip(y + error, 0.0, 1.0),
            color=colors[method],
            alpha=0.16,
        )

    axis.set_title(title)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.25)

for axis in axes[-1, :]:
    axis.set_xlabel(
        "Environment interactions (millions)"
    )

for axis in axes[:, 0]:
    axis.set_ylabel("Pass rate")

handles, legend_labels = (
    axes[0, 0].get_legend_handles_labels()
)

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=2,
)

fig.tight_layout(
    rect=(0.0, 0.0, 1.0, 0.94)
)

fig.savefig(
    CRITERIA_FIGURE,
    dpi=220,
)

plt.close(fig)


print()
print("===== Part B corrected development learning curves =====")

for iteration in ITERATIONS:
    central = next(
        row
        for row in method_curve_rows
        if (
            row["method"] == "central"
            and row["iteration"]
            == iteration
        )
    )

    dm2 = next(
        row
        for row in method_curve_rows
        if (
            row["method"] == "dm2"
            and row["iteration"]
            == iteration
        )
    )

    print(
        f"i{iteration:4d} "
        f"steps={central['interactions']:7d}  "
        f"Central LV="
        f"{central['lv_rate_mean']:.3f}"
        f"±{central['lv_rate_std']:.3f}  "
        f"DM2 LV="
        f"{dm2['lv_rate_mean']:.3f}"
        f"±{dm2['lv_rate_std']:.3f}  "
        f"DM2-Central="
        f"{dm2['lv_rate_mean'] - central['lv_rate_mean']:+.3f}"
    )


print()
print("===== Sample-efficiency result =====")

for method in ("central", "dm2"):
    result = results["methods"][method]
    crossing = result["first_95_percent_lv"]
    best = result["development_best_checkpoint"]
    final = result["final_checkpoint"]

    print()
    print(labels[method])

    if crossing is None:
        print(
            "  First 95% LV: not reached"
        )
    else:
        print(
            "  First 95% LV: "
            f"i{crossing['iteration']}, "
            f"{crossing['interactions']} "
            "interactions"
        )

    print(
        "  Normalized LV AUC: "
        f"{result['normalized_lv_auc']:.4f}"
    )

    print(
        "  Best development checkpoint: "
        f"i{best['iteration']}, "
        f"LV={best['lv_rate_mean']:.3f}"
        f"±{best['lv_rate_std']:.3f}"
    )

    print(
        "  Final checkpoint i2999: "
        f"LV={final['lv_rate_mean']:.3f}"
        f"±{final['lv_rate_std']:.3f}"
    )


print()
print("===== Top five development checkpoints =====")

for method in ("central", "dm2"):
    print()
    print(labels[method])

    top_rows = [
        row
        for row in ranking_rows
        if (
            row["method"] == method
            and row["rank"] <= 5
        )
    ]

    for row in top_rows:
        print(
            f"  {row['rank']}. "
            f"i{row['iteration']}: "
            f"LV={row['lv_rate_mean']:.3f}"
            f"±{row['lv_rate_std']:.3f}, "
            f"survival="
            f"{row['survival_rate_mean']:.3f}, "
            f"progress-pass="
            f"{row['progress_pass_rate_mean']:.3f}, "
            f"straight-pass="
            f"{row['straightness_pass_rate_mean']:.3f}, "
            f"CoT={row['path_cot_mean']:.3f}"
        )


print()
print(
    "Maximum survival-time cross-check "
    f"difference: "
    f"{maximum_survival_discrepancy:.6f} s"
)

print()
print("Saved:")
for output in (
    ROLLOUT_CSV,
    SEED_SUMMARY_CSV,
    METHOD_CURVE_CSV,
    RANKING_CSV,
    RESULT_JSON,
    LV_FIGURE,
    LV_FIGURE_PDF,
    CRITERIA_FIGURE,
):
    print(output)

print()
print(
    "PASS: Part B development learning curves "
    "use corrected and identical metrics for "
    "Central and DM2"
)

PASS_MARKER.write_text(
    "Part B development analysis passed\n"
    f"Development scenario seed: {SCENARIO_SEED}\n"
    f"Training seeds: 47,48,49\n"
    f"Evaluations: {2 * 3 * len(ITERATIONS)}\n"
    f"Rollouts: {2 * 3 * len(ITERATIONS) * NUM_ENVS}\n",
    encoding="utf-8",
)

print(f"Marker: {PASS_MARKER}")
