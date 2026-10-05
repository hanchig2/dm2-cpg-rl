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
watchdog_limit=180
watchdog_interval=10

echo "===== Part B remaining formal training ====="
echo "Start:  $(date --iso-8601=seconds)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"

if ! git merge-base \
  --is-ancestor \
  "$source_commit" \
  HEAD; then

    echo "ERROR: required Part B source commit is absent"
    exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: tracked working tree is dirty"
    git status --short
    exit 1
fi

if pgrep -af '[s]cripts/rsl_rl/train.py|[s]cripts/rsl_rl/eval.py' \
  >/dev/null; then

    echo "ERROR: another training/evaluation process is running"
    pgrep -af '[s]cripts/rsl_rl/train.py|[s]cripts/rsl_rl/eval.py'
    exit 1
fi

specifications=(
  "dm2|PartB-DM2-Message-CPG-Mixed-Unitree-A1-K1-v0|48"
  "central|PartB-Central-CPG-Mixed-Unitree-A1-K1-v0|49"
  "dm2|PartB-DM2-Message-CPG-Mixed-Unitree-A1-K1-v0|49"
)

run_training () {
    method="$1"
    task="$2"
    seed="$3"

    run_name="part_b_${method}_from_scratch_train${seed}_i3000"
    console_log="logs/rsl_rl/${run_name}.console.log"
    watchdog_log="analysis_outputs/part_b_from_scratch/${run_name}.watchdog.log"

    existing_run=$(
      find logs/rsl_rl \
        -mindepth 2 \
        -maxdepth 2 \
        -type d \
        -name "*${run_name}*" \
        -printf '%T@ %p\n' \
        2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )

    if [ -n "$existing_run" ] && \
       [ -s "$existing_run/model_2999.pt" ]; then

        echo "SKIP: already complete"
        echo "$existing_run/model_2999.pt"
        return 0
    fi

    if [ -n "$existing_run" ]; then
        echo "ERROR: an incomplete matching run already exists:"
        echo "$existing_run"
        echo "Inspect or preserve it before retrying."
        return 1
    fi

    echo
    echo "============================================================"
    echo "Part B formal training"
    echo "Method:      $method"
    echo "Task:        $task"
    echo "Train seed:  $seed"
    echo "Iterations:  3000"
    echo "Interactions: 9216000"
    echo "Watchdog:    ${watchdog_limit}s without log growth"
    echo "============================================================"

    : > "$console_log"
    : > "$watchdog_log"

    systemd-inhibit \
      --what=sleep:idle \
      --mode=block \
      --who="Part B ${method} seed ${seed}" \
      --why="Prevent sleep during formal training" \
      python -u scripts/rsl_rl/train.py \
        --task "$task" \
        --num_envs 64 \
        --seed "$seed" \
        --max_iterations 3000 \
        --run_name "$run_name" \
        --headless \
        > "$console_log" 2>&1 &

    inhibit_pid=$!

    last_size=0
    last_change=$(date +%s)
    watchdog_triggered=0

    while kill -0 "$inhibit_pid" 2>/dev/null; do
        sleep "$watchdog_interval"

        current_size=$(
          stat -c '%s' "$console_log" 2>/dev/null
        )

        current_size="${current_size:-0}"
        current_time=$(date +%s)

        if [ "$current_size" -gt "$last_size" ]; then
            last_size="$current_size"
            last_change="$current_time"
        fi

        no_change_seconds=$((current_time - last_change))

        latest_iteration=$(
          grep -oE \
            'Learning iteration +[0-9]+/[0-9]+' \
            "$console_log" \
            2>/dev/null \
            | tail -n 1
        )

        {
            echo "$(date --iso-8601=seconds)"
            echo "PID: $inhibit_pid"
            echo "Log bytes: $current_size"
            echo "No-change seconds: $no_change_seconds"
            echo "Iteration: ${latest_iteration:-initializing}"
            echo
        } >> "$watchdog_log"

        if [ "$no_change_seconds" -ge "$watchdog_limit" ]; then
            echo "ERROR: watchdog detected a probable stall"
            echo "No console-log growth for ${no_change_seconds}s"
            echo "Run: $run_name"

            mapfile -t train_pids < <(
              pgrep -f \
                "scripts/rsl_rl/train.py.*--run_name ${run_name}"
            )

            if [ "${#train_pids[@]}" -gt 0 ]; then
                echo "Sending SIGTERM to: ${train_pids[*]}"
                kill -TERM "${train_pids[@]}" 2>/dev/null
                sleep 15

                for pid in "${train_pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "Sending SIGKILL to: $pid"
                        kill -KILL "$pid" 2>/dev/null
                    fi
                done
            fi

            kill -TERM "$inhibit_pid" 2>/dev/null
            sleep 2
            kill -KILL "$inhibit_pid" 2>/dev/null

            watchdog_triggered=1
            break
        fi
    done

    wait "$inhibit_pid"
    process_status=$?

    echo "Process status: $process_status"
    echo "Watchdog triggered: $watchdog_triggered"

    if [ "$watchdog_triggered" -ne 0 ]; then
        echo "FAILED: probable training stall"
        tail -n 160 "$console_log"
        return 124
    fi

    if [ "$process_status" -ne 0 ]; then
        echo "FAILED: training process returned nonzero status"
        tail -n 200 "$console_log"
        return "$process_status"
    fi

    if grep -q \
      'Loading model checkpoint from' \
      "$console_log"; then

        echo "FAILED: formal from-scratch run loaded a checkpoint"
        return 1
    fi

    run_dir=$(
      find logs/rsl_rl \
        -mindepth 2 \
        -maxdepth 2 \
        -type d \
        -name "*${run_name}*" \
        -printf '%T@ %p\n' \
        2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )

    if [ -z "$run_dir" ] || \
       [ ! -s "$run_dir/model_2999.pt" ]; then

        echo "FAILED: model_2999.pt was not produced"
        echo "Run directory: ${run_dir:-missing}"
        return 1
    fi

    echo "PASS: $run_dir/model_2999.pt"
    return 0
}

completed=0
failed=0

for specification in "${specifications[@]}"; do
    method="${specification%%|*}"
    remainder="${specification#*|}"
    task="${remainder%%|*}"
    seed="${remainder##*|}"

    run_training "$method" "$task" "$seed"
    status=$?

    if [ "$status" -ne 0 ]; then
        failed=1
        echo "Stopping remaining jobs after failure."
        break
    fi

    completed=$((completed + 1))
done

echo
echo "===== Remaining-training result ====="
echo "Completed/skipped: $completed/3"
echo "Failed status:     $failed"
echo "Finish: $(date --iso-8601=seconds)"

if [ "$failed" -eq 0 ]; then
    echo "PASS: all remaining Part B training completed"
else
    echo "FAILED/INCOMPLETE: inspect logs before continuing"
fi

exit "$failed"
