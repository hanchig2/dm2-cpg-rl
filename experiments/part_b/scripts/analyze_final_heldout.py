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
from scipy.stats import binomtest


DT = 0.01
GRAVITY = 9.80665
MAX_STEPS = 3000
MAX_TIME = MAX_STEPS * DT

NUM_ENVS = 64
SCENARIO_SEEDS = (107, 108)
TRAIN_SEEDS = (47, 48, 49)

MIN_PROGRESS_RATIO = 0.70
MIN_STRAIGHTNESS = 0.80
MAX_ABS_YAW_BIAS = 0.05
MAX_TRANSLATION_RMSE = 0.30
TARGET_LV_RATE = 0.95

BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260928

LOCK_FILE = Path(
    "analysis_outputs/part_b_from_scratch/"
    "heldout_protocol/locked_checkpoints.tsv"
)

LOCK_MARKER = Path(
    "analysis_outputs/part_b_from_scratch/"
    "heldout_protocol/HELDOUT_PROTOCOL_LOCKED"
)

EVALUATION_MARKER = Path(
    "analysis_outputs/part_b_from_scratch/"
    "heldout_protocol/HELDOUT_EVALUATIONS_PASS"
)

ARTIFACT_AUDIT_MARKER = Path(
    "analysis_outputs/part_b_from_scratch/"
    "heldout_artifact_audit/HELDOUT_ARTIFACT_AUDIT_PASS"
)

OUTPUT_DIR = Path(
    "analysis_outputs/part_b_from_scratch/"
    "final_heldout_analysis"
)

ROLLOUT_CSV = OUTPUT_DIR / "heldout_rollouts_corrected.csv"
GROUP_CSV = OUTPUT_DIR / "heldout_cluster_summary.csv"
ROLE_CSV = OUTPUT_DIR / "heldout_role_summary.csv"

MATCHED_PAIR_CSV = (
    OUTPUT_DIR
    / "matched_budget_pairs.csv"
)

SELECTED_PAIR_CSV = (
    OUTPUT_DIR
    / "development_selected_pairs.csv"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "paired_comparison_summary.csv"
)

DECISION_JSON = (
    OUTPUT_DIR
    / "final_part_b_decision.json"
)

FIGURE_PATH = (
    OUTPUT_DIR
    / "heldout_lv_by_cluster.png"
)

PASS_MARKER = OUTPUT_DIR / "FINAL_HELDOUT_ANALYSIS_PASS"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if PASS_MARKER.exists():
    PASS_MARKER.unlink()


ROLE_ORDER = (
    "central_selected_and_matched",
    "dm2_matched_budget",
    "dm2_development_selected",
)

ROLE_LABELS = {
    "central_selected_and_matched": (
        "Central\ni2100"
    ),
    "dm2_matched_budget": (
        "DM²\ni2100"
    ),
    "dm2_development_selected": (
        "DM²\ni2300"
    ),
}


def read_tsv(path):
    with path.open(newline="") as input_file:
        return list(
            csv.DictReader(
                input_file,
                delimiter="\t",
            )
        )


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
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
            "Zero-norm quaternion."
        )

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def finite_mean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = np.isfinite(values)

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(values[valid])
    )


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


def summarize(rows):
    fields = (
        "survival_time",
        "translation_rmse",
        "abs_yaw_bias",
        "progress",
        "progress_ratio",
        "straightness",
        "path_cot",
    )

    result = {
        "num_rollouts": len(rows),
        "survival_rate": finite_mean(
            [
                row["survived"]
                for row in rows
            ]
        ),
        "progress_pass_rate": finite_mean(
            [
                row["progress_pass"]
                for row in rows
            ]
        ),
        "straightness_pass_rate": finite_mean(
            [
                row["straightness_pass"]
                for row in rows
            ]
        ),
        "yaw_pass_rate": finite_mean(
            [
                row["yaw_pass"]
                for row in rows
            ]
        ),
        "translation_pass_rate": finite_mean(
            [
                row["translation_pass"]
                for row in rows
            ]
        ),
        "balance_pass_rate": finite_mean(
            [
                row["balance_pass"]
                for row in rows
            ]
        ),
        "lv_rate": finite_mean(
            [
                row["lv_pass"]
                for row in rows
            ]
        ),
    }

    for field in fields:
        result[field] = finite_mean(
            [
                row[field]
                for row in rows
            ]
        )

    result["rmst"] = result.pop(
        "survival_time"
    )

    return result


def hierarchical_bootstrap(
    pair_rows,
    difference_field,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + sum(
            ord(character)
            for character
            in difference_field
        )
    )

    lookup = {}

    for train_seed in TRAIN_SEEDS:
        for scenario_seed in SCENARIO_SEEDS:
            cluster = sorted(
                [
                    row
                    for row in pair_rows
                    if (
                        row["train_seed"]
                        == train_seed
                        and row[
                            "scenario_seed"
                        ] == scenario_seed
                    )
                ],
                key=lambda row: (
                    row[
                        "environment_index"
                    ]
                ),
            )

            if len(cluster) != NUM_ENVS:
                raise RuntimeError(
                    f"Expected {NUM_ENVS} rows "
                    f"for cluster "
                    f"{train_seed}/"
                    f"{scenario_seed}, "
                    f"got {len(cluster)}."
                )

            lookup[
                (
                    train_seed,
                    scenario_seed,
                )
            ] = np.asarray(
                [
                    float(
                        row[
                            difference_field
                        ]
                    )
                    for row in cluster
                ],
                dtype=np.float64,
            )

    bootstrap_values = np.empty(
        BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):
        sampled_values = []

        sampled_train_seeds = rng.choice(
            TRAIN_SEEDS,
            size=len(TRAIN_SEEDS),
            replace=True,
        )

        for train_seed in (
            sampled_train_seeds
        ):
            sampled_scenarios = rng.choice(
                SCENARIO_SEEDS,
                size=len(SCENARIO_SEEDS),
                replace=True,
            )

            for scenario_seed in (
                sampled_scenarios
            ):
                cluster = lookup[
                    (
                        int(train_seed),
                        int(scenario_seed),
                    )
                ]

                environment_indices = (
                    rng.integers(
                        0,
                        NUM_ENVS,
                        size=NUM_ENVS,
                    )
                )

                sampled_values.extend(
                    cluster[
                        environment_indices
                    ]
                )

        bootstrap_values[
            replicate
        ] = float(
            np.mean(sampled_values)
        )

    lower, upper = np.percentile(
        bootstrap_values,
        (2.5, 97.5),
    )

    return {
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "probability_positive": float(
            np.mean(
                bootstrap_values > 0.0
            )
        ),
    }


for marker in (
    LOCK_MARKER,
    EVALUATION_MARKER,
    ARTIFACT_AUDIT_MARKER,
):
    if not marker.is_file():
        raise RuntimeError(
            f"Required marker missing: {marker}"
        )

locked_rows = read_tsv(LOCK_FILE)

if len(locked_rows) != 9:
    raise RuntimeError(
        f"Expected nine locked rows, "
        f"got {len(locked_rows)}."
    )

role_iterations = {}

for role in ROLE_ORDER:
    iterations = {
        int(row["iteration"])
        for row in locked_rows
        if row["role"] == role
    }

    if len(iterations) != 1:
        raise RuntimeError(
            f"Expected one common iteration for {role}, "
            f"got {sorted(iterations)}."
        )

    role_iterations[role] = next(iter(iterations))

ROLE_LABELS = {
    "central_selected_and_matched": (
        "Central\n"
        f"i{role_iterations['central_selected_and_matched']}"
    ),
    "dm2_matched_budget": (
        "DM²\n"
        f"i{role_iterations['dm2_matched_budget']}"
    ),
    "dm2_development_selected": (
        "DM²\n"
        f"i{role_iterations['dm2_development_selected']}"
    ),
}


rollout_rows = []
maximum_survival_discrepancy = 0.0


for locked in locked_rows:
    role = locked["role"]
    method = locked["method"]
    train_seed = int(
        locked["train_seed"]
    )
    iteration = int(
        locked["iteration"]
    )
    interactions = int(
        locked["interactions"]
    )

    checkpoint = Path(
        locked["checkpoint"]
    )

    run_dir = checkpoint.parent

    for scenario_seed in SCENARIO_SEEDS:
        eval_name = (
            f"part_b_final_heldout_"
            f"{role}_{method}_"
            f"train{train_seed}_"
            f"i{iteration}_"
            f"scenario{scenario_seed}_"
            f"n{NUM_ENVS}"
        )

        directory = (
            run_dir
            / "eval_logs"
            / eval_name
        )

        print(
            f"Analyzing {role}, "
            f"train seed {train_seed}, "
            f"scenario {scenario_seed}"
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

        robot_mass = float(
            np.asarray(
                load(
                    directory,
                    "robot_mass.npy",
                )
            ).reshape(())
        )

        expected_shapes = (
            (
                angular_velocity,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            (
                position,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            (
                quaternion,
                (NUM_ENVS, 4, MAX_STEPS),
            ),
            (
                tracked_velocity,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            (
                command,
                (NUM_ENVS, 3, MAX_STEPS),
            ),
            (
                energy,
                (NUM_ENVS, 1, MAX_STEPS),
            ),
            (
                survival_mask,
                (NUM_ENVS, 1, MAX_STEPS),
            ),
            (
                raw_survival_time,
                (NUM_ENVS,),
            ),
        )

        for value, expected_shape in (
            expected_shapes
        ):
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected shape "
                    f"{value.shape}; expected "
                    f"{expected_shape} in "
                    f"{directory}"
                )

        if (
            not math.isfinite(robot_mass)
            or robot_mass <= 0.0
        ):
            raise RuntimeError(
                f"Invalid robot mass: "
                f"{robot_mass}"
            )

        mask = (
            survival_mask[:, 0, :]
            > 0.5
        )

        survival_time = (
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
                    survival_time
                    - saved_survival_time
                )
            )
        )

        maximum_survival_discrepancy = max(
            maximum_survival_discrepancy,
            discrepancy,
        )

        if discrepancy > 2.1 * DT:
            raise RuntimeError(
                f"Survival-time mismatch: "
                f"{discrepancy}"
            )

        for environment_index in range(
            NUM_ENVS
        ):
            valid = np.flatnonzero(
                mask[environment_index]
            )

            if valid.size == 0:
                raise RuntimeError(
                    f"No valid samples in "
                    f"{directory}, env "
                    f"{environment_index}"
                )

            first_index = int(valid[0])

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

            error_xy = (
                actual_xy
                - target_xy
            )

            translation_rmse = math.sqrt(
                float(
                    np.mean(
                        np.sum(
                            error_xy
                            * error_xy,
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

            heading = np.asarray(
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
                    heading,
                )
            )

            if trajectory.shape[0] > 1:
                segments = np.diff(
                    trajectory,
                    axis=0,
                )

                path_length = float(
                    np.sum(
                        np.linalg.norm(
                            segments,
                            axis=1,
                        )
                    )
                )
            else:
                path_length = 0.0

            commanded_distance = float(
                np.sum(
                    np.linalg.norm(
                        target_xy,
                        axis=1,
                    )
                )
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
                    "role": role,
                    "method": method,
                    "train_seed": train_seed,
                    "scenario_seed": (
                        scenario_seed
                    ),
                    "iteration": iteration,
                    "interactions": (
                        interactions
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
                    "robot_mass": (
                        robot_mass
                    ),
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
    len(ROLE_ORDER)
    * len(TRAIN_SEEDS)
    * len(SCENARIO_SEEDS)
    * NUM_ENVS
)

if len(rollout_rows) != expected_rollouts:
    raise RuntimeError(
        f"Expected {expected_rollouts} "
        f"rollouts, got "
        f"{len(rollout_rows)}."
    )


cluster_groups = defaultdict(list)

for row in rollout_rows:
    cluster_groups[
        (
            row["role"],
            row["train_seed"],
            row["scenario_seed"],
        )
    ].append(row)


cluster_rows = []

for (
    role,
    train_seed,
    scenario_seed,
), rows in sorted(cluster_groups.items()):
    result = {
        "role": role,
        "method": rows[0]["method"],
        "iteration": rows[0]["iteration"],
        "interactions": rows[0][
            "interactions"
        ],
        "train_seed": train_seed,
        "scenario_seed": scenario_seed,
    }

    result.update(
        summarize(rows)
    )

    cluster_rows.append(result)


role_groups = defaultdict(list)

for row in rollout_rows:
    role_groups[
        row["role"]
    ].append(row)


role_rows = []

for role in ROLE_ORDER:
    rows = role_groups[role]

    if len(rows) != (
        len(TRAIN_SEEDS)
        * len(SCENARIO_SEEDS)
        * NUM_ENVS
    ):
        raise RuntimeError(
            f"Unexpected rollout count "
            f"for {role}: {len(rows)}"
        )

    result = {
        "role": role,
        "method": rows[0]["method"],
        "iteration": rows[0]["iteration"],
        "interactions": rows[0][
            "interactions"
        ],
    }

    result.update(
        summarize(rows)
    )

    role_clusters = [
        row
        for row in cluster_rows
        if row["role"] == role
    ]

    result["lv_cluster_std"] = (
        sample_std(
            [
                row["lv_rate"]
                for row in role_clusters
            ]
        )
    )

    result[
        "survival_cluster_std"
    ] = sample_std(
        [
            row["survival_rate"]
            for row in role_clusters
        ]
    )

    role_rows.append(result)


rollout_lookup = defaultdict(dict)

for row in rollout_rows:
    key = (
        row["train_seed"],
        row["scenario_seed"],
        row["environment_index"],
    )

    rollout_lookup[
        row["role"]
    ][key] = row


comparison_specs = (
    (
        "matched_budget",
        "central_selected_and_matched",
        "dm2_matched_budget",
    ),
    (
        "development_selected",
        "central_selected_and_matched",
        "dm2_development_selected",
    ),
)

metric_fields = (
    "lv_pass",
    "survival_time",
    "translation_rmse",
    "abs_yaw_bias",
    "progress",
    "progress_ratio",
    "straightness",
    "path_cot",
)

pair_outputs = {}
comparison_rows = []
decision_comparisons = {}


for (
    comparison_name,
    central_role,
    dm2_role,
) in comparison_specs:
    central_lookup = (
        rollout_lookup[central_role]
    )

    dm2_lookup = (
        rollout_lookup[dm2_role]
    )

    if set(central_lookup) != set(dm2_lookup):
        raise RuntimeError(
            f"Pair keys differ for "
            f"{comparison_name}."
        )

    pair_rows = []

    for key in sorted(central_lookup):
        central = central_lookup[key]
        dm2 = dm2_lookup[key]

        pair = {
            "comparison": comparison_name,
            "train_seed": key[0],
            "scenario_seed": key[1],
            "environment_index": key[2],
            "central_iteration": (
                central["iteration"]
            ),
            "dm2_iteration": (
                dm2["iteration"]
            ),
            "central_interactions": (
                central["interactions"]
            ),
            "dm2_interactions": (
                dm2["interactions"]
            ),
        }

        for metric in metric_fields:
            central_value = float(
                central[metric]
            )

            dm2_value = float(
                dm2[metric]
            )

            pair[
                f"central_{metric}"
            ] = central_value

            pair[
                f"dm2_{metric}"
            ] = dm2_value

            pair[
                f"difference_{metric}"
            ] = (
                dm2_value
                - central_value
            )

        pair_rows.append(pair)

    if len(pair_rows) != 384:
        raise RuntimeError(
            f"Expected 384 pairs for "
            f"{comparison_name}, got "
            f"{len(pair_rows)}."
        )

    pair_outputs[
        comparison_name
    ] = pair_rows

    both_pass = sum(
        int(
            row["central_lv_pass"] == 1
            and row["dm2_lv_pass"] == 1
        )
        for row in pair_rows
    )

    dm2_gains = sum(
        int(
            row["central_lv_pass"] == 0
            and row["dm2_lv_pass"] == 1
        )
        for row in pair_rows
    )

    dm2_losses = sum(
        int(
            row["central_lv_pass"] == 1
            and row["dm2_lv_pass"] == 0
        )
        for row in pair_rows
    )

    both_fail = sum(
        int(
            row["central_lv_pass"] == 0
            and row["dm2_lv_pass"] == 0
        )
        for row in pair_rows
    )

    discordant = (
        dm2_gains
        + dm2_losses
    )

    if discordant > 0:
        mcnemar_p = float(
            binomtest(
                dm2_gains,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        mcnemar_p = 1.0

    metric_results = {}

    for metric in metric_fields:
        difference_field = (
            f"difference_{metric}"
        )

        differences = np.asarray(
            [
                float(
                    row[difference_field]
                )
                for row in pair_rows
            ],
            dtype=np.float64,
        )

        finite = np.isfinite(
            differences
        )

        point_difference = float(
            np.mean(
                differences[finite]
            )
        )

        bootstrap = (
            hierarchical_bootstrap(
                pair_rows,
                difference_field,
            )
        )

        record = {
            "comparison": (
                comparison_name
            ),
            "metric": metric,
            "difference_definition": (
                "DM2 minus Central"
            ),
            "mean_difference": (
                point_difference
            ),
            "cluster_bootstrap_ci_lower": (
                bootstrap["ci_lower"]
            ),
            "cluster_bootstrap_ci_upper": (
                bootstrap["ci_upper"]
            ),
            "bootstrap_probability_positive": (
                bootstrap[
                    "probability_positive"
                ]
            ),
        }

        comparison_rows.append(record)
        metric_results[metric] = record

    decision_comparisons[
        comparison_name
    ] = {
        "both_pass": both_pass,
        "dm2_gains": dm2_gains,
        "dm2_losses": dm2_losses,
        "both_fail": both_fail,
        "mcnemar_exact_p_value": (
            mcnemar_p
        ),
        "metric_results": (
            metric_results
        ),
    }


write_csv(
    ROLLOUT_CSV,
    rollout_rows,
)

write_csv(
    GROUP_CSV,
    cluster_rows,
)

write_csv(
    ROLE_CSV,
    role_rows,
)

write_csv(
    MATCHED_PAIR_CSV,
    pair_outputs["matched_budget"],
)

write_csv(
    SELECTED_PAIR_CSV,
    pair_outputs[
        "development_selected"
    ],
)

write_csv(
    COMPARISON_CSV,
    comparison_rows,
)


role_by_name = {
    row["role"]: row
    for row in role_rows
}

decision = {
    "metric_definition": (
        "Corrected 2D-vector translation "
        "RMSE, initial-heading progress, "
        "path straightness, corrected "
        "survival units, scalar mass, and "
        "path-based CoT."
    ),
    "training_seeds": list(
        TRAIN_SEEDS
    ),
    "heldout_scenario_seeds": list(
        SCENARIO_SEEDS
    ),
    "num_environments_per_cluster": (
        NUM_ENVS
    ),
    "num_clusters_per_role": 6,
    "num_rollouts_per_role": 384,
    "bootstrap_replicates": (
        BOOTSTRAP_REPLICATES
    ),
    "maximum_survival_discrepancy_seconds": (
        maximum_survival_discrepancy
    ),
    "role_summaries": {
        role: role_by_name[role]
        for role in ROLE_ORDER
    },
    "comparisons": (
        decision_comparisons
    ),
    "caution": (
        "Only three independent training "
        "seeds were used. Rollout-level "
        "McNemar tests do not account for "
        "all within-model dependence; "
        "hierarchical cluster-bootstrap "
        "intervals are the primary "
        "uncertainty summary."
    ),
}

with DECISION_JSON.open("w") as output_file:
    json.dump(
        decision,
        output_file,
        indent=2,
        sort_keys=True,
    )


fig, axis = plt.subplots(
    figsize=(8.5, 5.8),
)

x_positions = np.arange(
    len(ROLE_ORDER),
    dtype=np.float64,
)

role_lv = [
    role_by_name[role]["lv_rate"]
    for role in ROLE_ORDER
]

colors = (
    "#1f77b4",
    "#d62728",
    "#ff7f0e",
)

axis.bar(
    x_positions,
    role_lv,
    color=colors,
    alpha=0.72,
    width=0.62,
)

rng = np.random.default_rng(47)

for role_index, role in enumerate(
    ROLE_ORDER
):
    cluster_values = [
        row["lv_rate"]
        for row in cluster_rows
        if row["role"] == role
    ]

    jitter = rng.uniform(
        -0.09,
        0.09,
        size=len(cluster_values),
    )

    axis.scatter(
        role_index + jitter,
        cluster_values,
        color="black",
        s=30,
        zorder=3,
    )

axis.axhline(
    TARGET_LV_RATE,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="95% LV target",
)

axis.set_xticks(
    x_positions,
    [
        ROLE_LABELS[role]
        for role in ROLE_ORDER
    ],
)

axis.set_ylabel(
    "Held-out Locomotion Viability rate"
)

axis.set_ylim(
    0.0,
    1.03,
)

axis.grid(
    axis="y",
    alpha=0.25,
)

axis.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    dpi=220,
)

plt.close(fig)


print()
print("===== Aggregate held-out metrics =====")
print(
    "Role                            "
    "Iter  Surv  RMST  TransRMSE "
    "AbsYaw Progress Ratio Straight "
    "CoT    LV"
)

for role in ROLE_ORDER:
    row = role_by_name[role]

    print(
        f"{role:31s} "
        f"{int(row['iteration']):4d} "
        f"{row['survival_rate']:5.3f} "
        f"{row['rmst']:5.2f} "
        f"{row['translation_rmse']:9.3f} "
        f"{row['abs_yaw_bias']:6.3f} "
        f"{row['progress']:8.2f} "
        f"{row['progress_ratio']:5.3f} "
        f"{row['straightness']:8.3f} "
        f"{row['path_cot']:5.3f} "
        f"{row['lv_rate']:5.3f}"
    )


print()
print("===== Criterion pass rates =====")
print(
    "Role                            "
    "Surv Progress Straight Yaw "
    "Translation Balance LV"
)

for role in ROLE_ORDER:
    row = role_by_name[role]

    print(
        f"{role:31s} "
        f"{row['survival_rate']:5.3f} "
        f"{row['progress_pass_rate']:8.3f} "
        f"{row['straightness_pass_rate']:8.3f} "
        f"{row['yaw_pass_rate']:5.3f} "
        f"{row['translation_pass_rate']:11.3f} "
        f"{row['balance_pass_rate']:7.3f} "
        f"{row['lv_rate']:5.3f}"
    )


for comparison_name in (
    "matched_budget",
    "development_selected",
):
    result = decision_comparisons[
        comparison_name
    ]

    print()
    print(
        "===== Paired comparison: "
        f"{comparison_name} ====="
    )

    print(
        f"Both pass:  {result['both_pass']}"
    )
    print(
        f"DM2 gains:  {result['dm2_gains']}"
    )
    print(
        f"DM2 losses: {result['dm2_losses']}"
    )
    print(
        f"Both fail:  {result['both_fail']}"
    )
    print(
        "McNemar exact p-value: "
        f"{result['mcnemar_exact_p_value']:.6f}"
    )

    print()
    print(
        "Mean paired difference "
        "(DM2 minus Central), "
        "hierarchical bootstrap 95% CI:"
    )

    labels = {
        "lv_pass": "LV rate",
        "survival_time": "Survival time",
        "translation_rmse": (
            "Translation RMSE"
        ),
        "abs_yaw_bias": "Abs yaw bias",
        "progress": "Progress",
        "progress_ratio": (
            "Progress ratio"
        ),
        "straightness": "Straightness",
        "path_cot": "Path CoT",
    }

    for metric in metric_fields:
        record = result[
            "metric_results"
        ][metric]

        print(
            f"{labels[metric]:20s} "
            f"{record['mean_difference']:+.4f} "
            f"["
            f"{record['cluster_bootstrap_ci_lower']:+.4f}, "
            f"{record['cluster_bootstrap_ci_upper']:+.4f}"
            f"]"
        )


print()
print("===== Final Part B evidence =====")

central_lv = role_by_name[
    "central_selected_and_matched"
]["lv_rate"]

dm2_matched_lv = role_by_name[
    "dm2_matched_budget"
]["lv_rate"]

dm2_selected_lv = role_by_name[
    "dm2_development_selected"
]["lv_rate"]

print(
    "Central i2100 held-out LV: "
    f"{central_lv:.3f}"
)

print(
    "DM2 i2100 held-out LV:     "
    f"{dm2_matched_lv:.3f}"
)

print(
    "DM2 i2300 held-out LV:     "
    f"{dm2_selected_lv:.3f}"
)

print(
    "Matched-budget difference: "
    f"{dm2_matched_lv - central_lv:+.3f}"
)

print(
    "Selected difference:       "
    f"{dm2_selected_lv - central_lv:+.3f}"
)

if central_lv >= TARGET_LV_RATE:
    print(
        "Central reaches the 95% held-out "
        "LV target."
    )
else:
    print(
        "Central does not reach the 95% "
        "held-out LV target."
    )

if dm2_selected_lv >= TARGET_LV_RATE:
    print(
        "DM2 reaches the 95% held-out "
        "LV target."
    )
else:
    print(
        "DM2 does not reach the 95% "
        "held-out LV target."
    )

if dm2_selected_lv > central_lv:
    print(
        "RESULT: development-selected DM2 "
        "improves held-out LV."
    )
else:
    print(
        "RESULT: development-selected DM2 "
        "does not improve held-out LV."
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
    GROUP_CSV,
    ROLE_CSV,
    MATCHED_PAIR_CSV,
    SELECTED_PAIR_CSV,
    COMPARISON_CSV,
    DECISION_JSON,
    FIGURE_PATH,
):
    print(output)

print()
print(
    "PASS: final Part B held-out analysis "
    "completed with locked checkpoints "
    "and untouched scenario seeds"
)

PASS_MARKER.write_text(
    "Part B final held-out analysis passed\n"
    f"Held-out scenario seeds: {','.join(map(str, SCENARIO_SEEDS))}\n"
    f"Training seeds: {','.join(map(str, TRAIN_SEEDS))}\n"
    f"Locked roles: {len(ROLE_ORDER)}\n"
    f"Rollouts: {len(ROLE_ORDER) * len(TRAIN_SEEDS) * len(SCENARIO_SEEDS) * NUM_ENVS}\n",
    encoding="utf-8",
)

print(f"Marker: {PASS_MARKER}")
