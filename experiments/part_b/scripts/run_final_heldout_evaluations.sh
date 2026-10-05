#!/usr/bin/env bash

set +e

script_dir="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" \
    && pwd
)"

repo_root="$(
  git -C "$script_dir" \
    rev-parse --show-toplevel \
    2>/dev/null
)" || {
    echo "ERROR: script is not inside a Git worktree"
    exit 1
}

cd "$repo_root" || {
    echo "ERROR: cannot enter repository root"
    exit 1
}

if [ "${CONDA_DEFAULT_ENV:-}" != "isaaclab" ]; then
    echo "ERROR: activate the required environment first:"
    echo "conda activate isaaclab"
    exit 1
fi

export PYTHONPATH="$repo_root:$repo_root/exts/cpg:${PYTHONPATH:-}"

mkdir -p "logs/rsl_rl"

source_commit="9ae10d2d8e2c5760b41439f4b9d42a190ee2eacf"

lock_file="analysis_outputs/part_b_from_scratch/heldout_protocol/locked_checkpoints.tsv"
lock_marker="analysis_outputs/part_b_from_scratch/heldout_protocol/HELDOUT_PROTOCOL_LOCKED"
pass_marker="analysis_outputs/part_b_from_scratch/heldout_protocol/HELDOUT_EVALUATIONS_PASS"

scenario_seeds=(
  "107"
  "108"
)

num_envs=64
timeout_seconds=1200
total=18

required_artifacts=(
  "base_angular_velocity.npy"
  "base_position.npy"
  "base_quaternion.npy"
  "base_tracked_velocity.npy"
  "command.npy"
  "death_status.npy"
  "energy.npy"
  "feet_contacts.npy"
  "feet_forces.npy"
  "feet_position.npy"
  "hip_position_z.npy"
  "joint_acceleration.npy"
  "joint_position.npy"
  "robot_mass.npy"
  "survival_mask.npy"
  "survival_time.npy"
)

artifact_set_valid () {
    directory="$1"

    if [ ! -d "$directory" ]; then
        return 1
    fi

    for filename in "${required_artifacts[@]}"; do
        if [ ! -s "$directory/$filename" ]; then
            return 1
        fi
    done

    return 0
}

echo "===== Part B final held-out evaluations ====="
echo "Start:         $(date --iso-8601=seconds)"
echo "Branch:        $(git branch --show-current)"
echo "Commit:        $(git rev-parse HEAD)"
echo "Scenario seeds: ${scenario_seeds[*]}"
echo "Environments:  $num_envs"
echo "Evaluations:   $total"

failed=0

if ! git merge-base \
  --is-ancestor \
  "$source_commit" \
  HEAD; then

    echo "ERROR: required Part B source commit is absent"
    failed=1

elif [ ! -s "$lock_file" ]; then
    echo "ERROR: locked-checkpoint file missing"
    failed=1

elif [ ! -s "$lock_marker" ]; then
    echo "ERROR: held-out protocol lock marker missing"
    failed=1

elif [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: tracked working tree is dirty"
    git status --short
    failed=1

elif pgrep -af \
  '[s]cripts/rsl_rl/train.py|[s]cripts/rsl_rl/eval.py' \
  >/dev/null; then

    echo "ERROR: another training/evaluation process is running"

    pgrep -af \
      '[s]cripts/rsl_rl/train.py|[s]cripts/rsl_rl/eval.py'

    failed=1
fi

locked_count=$(
  tail -n +2 "$lock_file" \
    | wc -l
)

if [ "$locked_count" -ne 9 ]; then
    echo "ERROR: expected nine locked checkpoints"
    echo "Found: $locked_count"
    failed=1
fi

available_kib=$(
  df -Pk . \
    | awk 'NR == 2 {print $4}'
)

available_kib="${available_kib:-0}"
required_kib=$((5 * 1024 * 1024))

echo "Available disk: $((available_kib / 1024 / 1024)) GiB"

if [ "$available_kib" -lt "$required_kib" ]; then
    echo "ERROR: less than 5 GiB available"
    failed=1
fi

completed=0
skipped=0

if [ "$failed" -eq 0 ]; then
    while IFS=$'\t' read -r \
      role \
      method \
      train_seed \
      iteration \
      interactions \
      dev_lv_mean \
      dev_lv_std \
      checkpoint \
      expected_hash \
      selection_reason
    do
        if [ "$role" = "role" ]; then
            continue
        fi

        case "$method" in
            central)
                task="PartB-Central-CPG-Rough-Unitree-A1-K1-Eval-Moderate-v0"
                ;;
            dm2)
                task="PartB-DM2-Message-CPG-Rough-Unitree-A1-K1-Eval-Moderate-v0"
                ;;
            *)
                echo "ERROR: unknown method: $method"
                failed=1
                break
                ;;
        esac

        if [ ! -s "$checkpoint" ]; then
            echo "ERROR: locked checkpoint missing"
            echo "$checkpoint"
            failed=1
            break
        fi

        actual_hash=$(
          sha256sum "$checkpoint" \
            | cut -d' ' -f1
        )

        if [ "$actual_hash" != "$expected_hash" ]; then
            echo "ERROR: locked checkpoint hash changed"
            echo "$checkpoint"
            echo "Expected: $expected_hash"
            echo "Actual:   $actual_hash"
            failed=1
            break
        fi

        run_dir=$(dirname "$checkpoint")

        for scenario_seed in "${scenario_seeds[@]}"; do
            eval_name="part_b_final_heldout_${role}_${method}_train${train_seed}_i${iteration}_scenario${scenario_seed}_n${num_envs}"

            artifact_dir="$run_dir/eval_logs/$eval_name"
            console_log="logs/rsl_rl/${eval_name}.console.log"

            if artifact_set_valid "$artifact_dir"; then
                echo
                echo "SKIP: $role seed $train_seed scenario $scenario_seed"
                skipped=$((skipped + 1))
                continue
            fi

            if [ -e "$artifact_dir" ]; then
                echo "ERROR: incomplete artifact directory exists"
                echo "$artifact_dir"
                failed=1
                break
            fi

            accounted=$((completed + skipped))

            echo
            echo "============================================================"
            echo "Part B final held-out evaluation"
            echo "Role:            $role"
            echo "Method:          $method"
            echo "Training seed:   $train_seed"
            echo "Checkpoint:      model_${iteration}.pt"
            echo "Interactions:    $interactions"
            echo "Scenario seed:   $scenario_seed"
            echo "Evaluation envs: $num_envs"
            echo "Accounted for:   $accounted/$total"
            echo "============================================================"

            timeout \
              --signal=TERM \
              --kill-after=20s \
              "${timeout_seconds}s" \
              systemd-inhibit \
                --what=sleep:idle \
                --mode=block \
                --who="Part B heldout ${role} seed ${train_seed}" \
                --why="Prevent sleep during held-out evaluation" \
                python -u scripts/rsl_rl/eval.py \
                  --task "$task" \
                  --num_envs "$num_envs" \
                  --eval_seed "$scenario_seed" \
                  --checkpoint "$checkpoint" \
                  --eval_name "$eval_name" \
                  --save_eval_logs \
                  --headless \
                  > "$console_log" 2>&1

            process_status=$?

            if [ "$process_status" -ne 0 ]; then
                echo "ERROR: held-out evaluation failed"
                echo "Process status: $process_status"
                echo "Console log:    $console_log"

                tail -n 220 "$console_log"

                failed=1
                break
            fi

            if ! artifact_set_valid "$artifact_dir"; then
                echo "ERROR: held-out artifacts incomplete"
                echo "$artifact_dir"
                failed=1
                break
            fi

            artifact_count=$(
              find "$artifact_dir" \
                -maxdepth 1 \
                -type f \
                -name '*.npy' \
                | wc -l
            )

            completed=$((completed + 1))

            echo "PASS: $role seed $train_seed scenario $scenario_seed"
            echo "Saved NumPy artifacts: $artifact_count"
        done

        if [ "$failed" -ne 0 ]; then
            break
        fi
    done < "$lock_file"
fi

accounted=$((completed + skipped))

echo
echo "===== Final held-out evaluation result ====="
echo "Completed this launch: $completed"
echo "Skipped existing:      $skipped"
echo "Accounted for:         $accounted/$total"
echo "Failed status:         $failed"
echo "Finish: $(date --iso-8601=seconds)"

if [ "$failed" -eq 0 ] && \
   [ "$accounted" -eq "$total" ]; then

    date --iso-8601=seconds \
      > "$pass_marker"

    echo "PASS: all 18 held-out evaluations completed"
    echo "NEXT: calculate paired held-out statistics"
else
    echo "FAILED/INCOMPLETE: inspect the reported evaluation"
fi

exit "$failed"
