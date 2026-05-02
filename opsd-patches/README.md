# OPSD repo patches — how to apply

These are the OPSD-repo script modifications and new wrapper scripts made during the project. They turn hardcoded behaviors into env-var-driven knobs (so we can override at launch time) and add Stage 1 wrappers.

## On the new machine

```bash
# 1. Clone the upstream OPSD repo
git clone https://github.com/HJSang/OPSD_OnPolicyDistillation.git ~/OPSD_OnPolicyDistillation

# 2. Apply our patches by overlaying these files
cp -rv opsd-patches/scripts/* ~/OPSD_OnPolicyDistillation/scripts/

# 3. Make the new wrappers executable (cp may have lost +x)
chmod +x ~/OPSD_OnPolicyDistillation/scripts/opd/run_smoke.sh \
         ~/OPSD_OnPolicyDistillation/scripts/opd/run_stage1_b2.sh \
         ~/OPSD_OnPolicyDistillation/scripts/grpo/run_stage1_b1.sh
```

## What was changed

### `scripts/opd/train_opd.sh`

1. `GPUS_PER_NODE` is now env-overridable:
   ```bash
   GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
   ```
   Useful when you want to constrain to a subset of GPUs (with `CUDA_VISIBLE_DEVICES`) without the script auto-detecting all of them.

2. Added `ROLLOUT_NAME` env override (default still `sglang`):
   ```bash
   rollout_name=${ROLLOUT_NAME:-sglang}
   ...
   actor_rollout_ref.rollout.name=${rollout_name} \
   ```
   The original script hardcoded `name=sglang`. If sglang isn't installed (e.g. on machines with only vllm), set `ROLLOUT_NAME=vllm`.

3. Added `OUTPUT_DIR` env override (was hardcoded `${DATA_DIR}/grpo_processed`):
   ```bash
   OUTPUT_DIR=${OUTPUT_DIR:-${DATA_DIR}/grpo_processed}
   ```

4. Added `SKIP_PREPARE` env flag to skip re-running `prepare_grpo_data.py` if data is already prepared:
   ```bash
   if [ -z "${SKIP_PREPARE}" ]; then
       python "${SRC_ROOT}/data/prepare_grpo_data.py" ...
   else
       echo "SKIP_PREPARE=1 - using existing data in $OUTPUT_DIR"
   fi
   ```

### `scripts/eval/eval_math.sh`, `scripts/grpo/train_grpo.sh`, `scripts/grpo/train_grpo_native.sh`

Same `ROLLOUT_NAME` patch applied to each:
```bash
actor_rollout_ref.rollout.name=${ROLLOUT_NAME:-sglang} \
```

### New files

- **`scripts/opd/run_smoke.sh`** — Stage 0 smoke-test wrapper. Tiny data slice, 2 epochs, vllm rollout, 8192 → 2048 max_response. Validates pipeline end-to-end in ~5 min.
- **`scripts/opd/run_stage1_b2.sh`** — Stage 1 single-teacher OPD on math wrapper. Defaults: 3 epochs, bs 256, max_response 8192, vllm. **NOTE: defaults still point to Qwen3 paths from prior machine — update `MODEL_PATH` and `TEACHER_MODEL_PATH` for the Qwen2.5 strategy before running.**
- **`scripts/grpo/run_stage1_b1.sh`** — Stage 1 GRPO baseline wrapper. Pure RL, no teacher. **Same caveat: update model paths.**

## Why these changes were made

| Change | Reason |
|---|---|
| `ROLLOUT_NAME` env | sglang wasn't installed on the prior machine; needed to use vllm without modifying repo's main code each time |
| `OUTPUT_DIR` / `SKIP_PREPARE` | Wanted to reuse pre-built smoke-test data slice without re-running prepare_grpo_data.py every time |
| `GPUS_PER_NODE` env | When another process on the machine occupies some GPUs (resource contention), need to constrain ourselves to e.g. 2 GPUs without the script auto-grabbing all 8 |
| `run_*.sh` wrappers | Hide the repetitive env exports and remember the launch config for each Stage 1 baseline |

## Caveats

- The `run_stage1_b1.sh` / `run_stage1_b2.sh` defaults still reference Qwen3 paths (`/home/ubuntu/models/Qwen3-1.7B-Base`, `/home/ubuntu/models/Qwen3-8B-Base-GRPO-DAPO`). After downloading Qwen2.5 models on the new machine, either edit those defaults or always pass overrides at launch.
- TP_SIZE=2 was observed to fail on the prior A100 setup (`custom_all_reduce.cuh:455 'invalid argument'`). Use TP_SIZE=1 unless verified otherwise on the new machine.
- vllm 0.19.0 has a `cumem_allocator` OOM issue after long training runs (sleep/wake cycles accumulate fragmentation). Lowering `gpu_memory_utilization` to 0.5 from 0.6 can help.
