"""Test deepspeed init_distributed alone (no zero.init, no HF model)."""
import os
import torch

def main():
    print(f"rank={os.environ.get('RANK','?')} torch={torch.__version__}", flush=True)
    import deepspeed
    print(f"deepspeed {deepspeed.__version__}", flush=True)

    print("calling deepspeed.init_distributed()...", flush=True)
    deepspeed.init_distributed()
    print(f"rank={torch.distributed.get_rank()}/{torch.distributed.get_world_size()} init_distributed OK", flush=True)

    print("now activating HfDeepSpeedConfig (zero.init context)...", flush=True)
    from transformers.integrations import HfDeepSpeedConfig
    ds_cfg = {
        "train_batch_size": 8,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 3},
    }
    dschf = HfDeepSpeedConfig(ds_cfg)
    print("HfDeepSpeedConfig OK", flush=True)

    print("now loading model under zero.init...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"model loaded {type(model).__name__}", flush=True)


if __name__ == "__main__":
    main()
