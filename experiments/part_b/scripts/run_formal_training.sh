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
marker="analysis_outputs/part_b_from_scratch/audit/SMOKE_AUDIT_PASS"

echo "===== Part B formal paired training ====="
echo "Start: $(date --iso-8601=seconds)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"

failed=0

if ! git merge-base \
  --is-ancestor \
  "$source_commit" \
  HEAD; then

    echo "ERROR: required Part B source commit is absent"
    failed=1

elif [ ! -s "$marker" ]; then
    echo "ERROR: smoke-audit marker missing"
    failed=1

elif [ -n "$(
  git status --porcelain --untracked-files=no
)" ]; then
    echo "ERROR: tracked working tree is dirty"
    git status --short
    failed=1
fi

specifications=(
  "central|PartB-Central-CPG-Mixed-Unitree-A1-K1-v0|47"
  "dm2|PartB-DM2-Message-CPG-Mixed-Unitree-A1-K1-v0|47"
  "central|PartB-Central-CPG-Mixed-Unitree-A1-K1-v0|48"
  "dm2|PartB-DM2-Message-CPG-Mixed-Unitree-A1-K1-v0|48"
  "central|PartB-Central-CPG-Mixed-Unitree-A1-K1-v0|49"
  "dm2|PartB-DM2-Message-CPG-Mixed-Unitree-A1-K1-v0|49"
)

completed=0
skipped=0

if [ "$failed" -eq 0 ]; then
    for specification in "${specifications[@]}"; do
        remainder="${specification#*|}"
        slug="${specification%%|*}"
        task="${remainder%%|*}"
        seed="${remainder##*|}"

        run_name="part_b_${slug}_from_scratch_train${seed}_i3000"
        console_log="logs/rsl_rl/${run_name}.console.log"

        mapfile -t matches < <(
          find \
            logs/rsl_rl \
            -mindepth 2 \
            -maxdepth 2 \
            -type d \
            -name "*${run_name}*" \
            -print \
            2>/dev/null
        )

        complete_directory=""

        for directory in "${matches[@]}"; do
            if [ -s "$directory/model_2999.pt" ]; then
                complete_directory="$directory"
                break
            fi
        done

        if [ -n "$complete_directory" ]; then
            echo
            echo "SKIP: already complete"
            echo "$complete_directory"
            skipped=$((skipped + 1))
            continue
        fi

        if [ "${#matches[@]}" -gt 0 ]; then
            echo
            echo "ERROR: incomplete prior run exists:"
            printf '%s\n' "${matches[@]}"
            echo "Stop for manual inspection."
            failed=1
            break
        fi

        echo
        echo "============================================================"
        echo "Formal Part B training"
        echo "Method:      $slug"
        echo "Task:        $task"
        echo "Train seed:  $seed"
        echo "Iterations:  3000"
        echo "Interactions: 9216000"
        echo "Completed:   $completed/6"
        echo "============================================================"

        systemd-inhibit \
          --what=sleep:idle \
          --mode=block \
          --who="Part B ${slug} seed ${seed}" \
          --why="Prevent sleep during formal training" \
          python -u scripts/rsl_rl/train.py \
            --task "$task" \
            --num_envs 64 \
            --seed "$seed" \
            --max_iterations 3000 \
            --run_name "$run_name" \
            --headless \
            > "$console_log" 2>&1

        process_status=$?

        echo "Process status: $process_status"

        if [ "$process_status" -ne 0 ]; then
            echo "ERROR: training failed"
            tail -n 240 "$console_log"
            failed=1
            break
        fi

        if grep -q \
          'Loading model checkpoint from' \
          "$console_log"; then

            echo "ERROR: formal run loaded a checkpoint"
            grep -n \
              'Loading model checkpoint from' \
              "$console_log"
            failed=1
            break
        fi

        run_directory=$(
          find \
            logs/rsl_rl \
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

        if [ -z "$run_directory" ] || \
           [ ! -s "$run_directory/model_2999.pt" ]; then

            echo "ERROR: final checkpoint missing"
            echo "Run directory: $run_directory"
            failed=1
            break
        fi

        echo "PASS: $run_directory/model_2999.pt"

        completed=$((completed + 1))
    done
fi

echo
echo "===== Formal training result ====="
echo "Completed this launch: $completed"
echo "Skipped complete:      $skipped"
echo "Failed status:         $failed"
echo "Finish: $(date --iso-8601=seconds)"

if [ "$failed" -eq 0 ] && \
   [ $((completed + skipped)) -eq 6 ]; then

    echo "PASS: all six Part B training runs completed"
    echo "NEXT: paired checkpoint evaluation"
else
    echo "FAILED/INCOMPLETE: inspect logs before continuing"
fi
