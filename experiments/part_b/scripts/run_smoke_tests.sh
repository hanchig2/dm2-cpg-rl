#!/usr/bin/env bash

set +e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)" || exit 1
cd "$repo_root" || exit 1

if [ "${CONDA_DEFAULT_ENV:-}" != "isaaclab" ]; then
    echo "ERROR: activate the isaaclab environment first"
    exit 1
fi

export PYTHONPATH="$repo_root:$repo_root/exts/cpg:${PYTHONPATH:-}"

mkdir -p "logs/rsl_rl"

source_commit="9ae10d2d8e2c5760b41439f4b9d42a190ee2eacf"
output_dir="analysis_outputs/part_b_from_scratch/audit"
marker="$output_dir/SMOKE_AUDIT_PASS"
manifest="$output_dir/smoke_runs.tsv"

mkdir -p "$output_dir"
rm -f "$marker"

if ! git merge-base --is-ancestor "$source_commit" HEAD; then
    echo "ERROR: required Part B source commit is absent"
    exit 1
fi

if pgrep -af '[s]cripts/rsl_rl/train.py|[s]cripts/rsl_rl/eval.py' >/dev/null; then
    echo "ERROR: another training/evaluation process is running"
    pgrep -af '[s]cripts/rsl_rl/train.py|[s]cripts/rsl_rl/eval.py'
    exit 1
fi

printf 'method\ttask\texperiment\trun_name\trun_dir\tconsole_log\tstatus\n' > "$manifest"

specifications=(
  "central|PartB-Central-CPG-Mixed-Unitree-A1-K1-v0|part_b_central_from_scratch_static_mixed_cpg_unitree_a1"
  "dm2|PartB-DM2-Message-CPG-Mixed-Unitree-A1-K1-v0|part_b_dm2_message_from_scratch_static_mixed_cpg_unitree_a1"
)

failed=0

for specification in "${specifications[@]}"; do
    method="${specification%%|*}"
    remainder="${specification#*|}"
    task="${remainder%%|*}"
    experiment="${remainder##*|}"
    run_name="part_b_${method}_repro_smoke_seed47_i2"
    console_log="logs/rsl_rl/${run_name}.console.log"

    echo
    echo "===== Smoke training: $method ====="

    systemd-inhibit \
      --what=sleep:idle \
      --mode=block \
      --who="Part B ${method} reproducibility smoke" \
      --why="Prevent sleep during smoke test" \
      python -u scripts/rsl_rl/train.py \
        --task "$task" \
        --num_envs 32 \
        --seed 47 \
        --max_iterations 2 \
        --run_name "$run_name" \
        --headless \
        > "$console_log" 2>&1

    process_status=$?

    run_dir="$(
      find "logs/rsl_rl/$experiment" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name "*_${run_name}" \
        -printf '%T@ %p\n' \
        2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"

    status="FAIL"

    if [ "$process_status" -eq 0 ] && \
       [ -n "$run_dir" ] && \
       [ -s "$run_dir/model_1.pt" ] && \
       [ -s "$run_dir/params/env.yaml" ] && \
       [ -s "$run_dir/params/agent.yaml" ] && \
       grep -q 'Learning iteration 1/2' "$console_log" && \
       ! grep -q 'Loading model checkpoint from' "$console_log"; then

        python - "$run_dir/model_1.pt" <<'PY_CHECKPOINT'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)

if int(checkpoint.get("iter", -1)) != 1:
    raise RuntimeError("Unexpected checkpoint iteration")

state = checkpoint.get("model_state_dict")
if not isinstance(state, dict) or not state:
    raise RuntimeError("Missing model_state_dict")

bad = [
    key
    for key, value in state.items()
    if torch.is_tensor(value)
    and (value.is_floating_point() or value.is_complex())
    and not torch.isfinite(value).all()
]

if bad:
    raise RuntimeError(f"Non-finite model tensors: {bad}")
PY_CHECKPOINT

        if [ "$?" -eq 0 ]; then
            status="PASS"
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$method" \
      "$task" \
      "$experiment" \
      "$run_name" \
      "$run_dir" \
      "$console_log" \
      "$status" \
      >> "$manifest"

    if [ "$status" != "PASS" ]; then
        echo "ERROR: $method smoke test failed"
        tail -n 160 "$console_log" 2>/dev/null
        failed=1
        break
    fi

    echo "PASS: $method smoke test"
done

if [ "$failed" -eq 0 ] && [ "$(grep -c $'\tPASS$' "$manifest")" -eq 2 ]; then
    {
        echo "Part B paired smoke audit passed"
        echo "Methods: central,dm2"
        echo "Training seed: 47"
        echo "Environments: 32"
        echo "Iterations: 2"
        echo "Source commit: $source_commit"
    } > "$marker"
    echo "PASS: paired smoke tests completed"
else
    echo "FAILED: smoke marker was not created"
    exit 1
fi
