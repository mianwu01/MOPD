"""Minimal deepspeed zero3 init test — replicates open-r1 GRPO model load."""
import os
import torch

def main():
    import deepspeed
    print(f"torch {torch.__version__}, deepspeed {deepspeed.__version__}", flush=True)
    print(f"rank={os.environ.get('RANK','?')}, local_rank={os.environ.get('LOCAL_RANK','?')}", flush=True)

    # Activate zero.init context (HF transformers does this automatically when
    # ZeRO-3 is detected via deepspeed config)
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.integrations import HfDeepSpeedConfig

    ds_cfg = {
        "train_batch_size": 8,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }
    print("creating HfDeepSpeedConfig (activates zero.init)...", flush=True)
    dschf = HfDeepSpeedConfig(ds_cfg)

    # Init deepspeed distributed
    deepspeed.init_distributed()
    print(f"after init_distributed: world={torch.distributed.get_world_size()}, rank={torch.distributed.get_rank()}", flush=True)

    attn_impl = "sdpa"   # hardcoded for diag — flash_attn fails under zero.init
    print(f"loading Qwen2.5-1.5B-Instruct (attn={attn_impl}) under zero.init...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    print(f"model loaded OK, type={type(model).__name__}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
