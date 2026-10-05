# Part B reproducibility package

This directory reproduces the paired from-scratch comparison between the
centralized CPG-RL baseline and DM²-CPG-RL. It contains executable launchers,
artifact audits, checkpoint-selection logic, held-out evaluation, and the exact
analysis scripts used for the reported results.

Large generated files are intentionally excluded from Git: model checkpoints,
NumPy rollout arrays, TensorBoard data, and console logs are regenerated
locally under `logs/rsl_rl/` and `analysis_outputs/part_b_from_scratch/`.

## Frozen protocol

| Item | Value |
|---|---|
| Training methods | Centralized CPG-RL, DM²-CPG-RL |
| Training seeds | 47, 48, 49 |
| Environments per training run | 64 |
| Training iterations | 3000 |
| Interactions per run | 9,216,000 |
| Development scenario seed | 101 |
| Development checkpoints | 0, 100, ..., 2900, 2999 |
| Development evaluations | 186 |
| Held-out scenario seeds | 107, 108 |
| Checkpoint selection | Aggregate development results; never per seed |
| Core source commit | `9ae10d2d8e2c5760b41439f4b9d42a190ee2eacf` |

The packaged launchers accept descendants of the core source commit so that
documentation and reproducibility scripts can be committed afterward without
invalidating the experiment source.

## Environment

Use Ubuntu 22.04, Isaac Sim 4.5, Isaac Lab 2.1, Python 3.10, PyTorch 2.5.1,
RSL-RL 2.3.1, and the repository's `isaaclab` Conda environment. The analysis
also uses NumPy, Matplotlib, SciPy, and PyYAML. See
`requirements-analysis.txt` for the direct analysis imports.

Activate the environment and enter the repository root before every stage:

```bash
conda activate isaaclab
cd /path/to/visual_postural_cpgrl
export PYTHONPATH="$PWD:$PWD/exts/cpg:${PYTHONPATH:-}"
```

Do not use held-out seeds 107 or 108 before the protocol-lock stage.

## End-to-end execution order

### 1. Paired smoke tests

```bash
bash experiments/part_b/scripts/run_smoke_tests.sh
```

This creates `analysis_outputs/part_b_from_scratch/audit/SMOKE_AUDIT_PASS`.

### 2. Six formal from-scratch training runs

```bash
nohup bash experiments/part_b/scripts/run_formal_training.sh \
  > analysis_outputs/part_b_from_scratch/formal_training.master.log \
  2>&1 &
```

The launcher refuses to overwrite incomplete matching runs. The optional
`scripts/recovery/run_remaining_training.sh` is retained for inspecting and
continuing the historical staged workflow; it does not silently resume a
checkpoint.

### 3. Audit training and generate the local run manifest

```bash
python experiments/part_b/scripts/audit_formal_training.py
```

The audit loads all 186 saved checkpoints, validates tensor finiteness and
layout consistency, confirms from-scratch console evidence, hashes endpoints,
and creates `formal_run_manifest.tsv` using repository-relative paths.

### 4. Development checkpoint sweep

```bash
nohup bash experiments/part_b/scripts/run_dev_checkpoint_sweep.sh \
  > analysis_outputs/part_b_from_scratch/development_sweep.master.log \
  2>&1 &
```

This evaluates both methods, all three training seeds, and all 31 checkpoints
with development scenario seed 101: 186 evaluations total.

### 5. Audit development arrays

```bash
python experiments/part_b/scripts/audit_dev_artifacts.py
```

### 6. Compute learning curves and sample efficiency

```bash
python experiments/part_b/scripts/analyze_dev_learning_curves.py \
  2>&1 | tee analysis_outputs/part_b_from_scratch/dev_analysis.log
```

The selection thresholds are defined directly in the analysis script:

- progress ratio at least 0.70;
- straightness at least 0.80;
- absolute yaw bias at most 0.05 rad/s;
- translational RMSE at most 0.30 m/s;
- target locomotion viability rate 0.95.

### 7. Lock held-out checkpoints

```bash
python experiments/part_b/scripts/lock_heldout_protocol.py
```

This verifies that held-out directories do not yet exist, applies the aggregate
development-selection policy, hashes all nine locked checkpoints, and writes
repository-relative paths to `locked_checkpoints.tsv`.

### 8. Run the 18 held-out evaluations

```bash
nohup bash experiments/part_b/scripts/run_final_heldout_evaluations.sh \
  > analysis_outputs/part_b_from_scratch/heldout_evaluations.master.log \
  2>&1 &
```

### 9. Audit held-out arrays

```bash
python experiments/part_b/scripts/audit_heldout_artifacts.py
```

### 10. Run final held-out analysis

```bash
python experiments/part_b/scripts/analyze_final_heldout.py \
  2>&1 | tee analysis_outputs/part_b_from_scratch/final_heldout_analysis.log
```

## Metric definitions

Both methods use exactly the same trajectory calculations and pass criteria.
The development and held-out scripts calculate translational tracking RMSE,
absolute yaw-rate bias, forward progress relative to the initial heading,
commanded-distance-normalized progress, path straightness, path cost of
transport, survival, balance, and locomotion viability from the saved arrays.

## Reference results

`reference/` contains compact, human-readable summaries from the original run.
They are evidence for comparison, not runtime inputs. A fresh reproduction
generates its own manifest, checkpoint hashes, protocol lock, curves, figures,
and final decision under `analysis_outputs/part_b_from_scratch/`.

The original development selection was Central iteration 2100 and DM²
iteration 2300, with DM² iteration 2100 additionally locked for the strict
matched-interaction comparison. The locking script recomputes these choices
from the newly generated development results rather than hard-coding them.

## Safety and provenance

- No checkpoint, raw rollout, or machine-specific absolute path is committed.
- Every evaluation stage is resumable by skipping only complete artifact sets.
- Audit markers are generated only after structural and finite-value checks.
- Held-out checkpoint selection is completed and recorded before seeds 107 and
  108 are evaluated.
- Push this branch to the project owner's publication remote, not the upstream
  repository, unless upstream maintainers explicitly request it.
