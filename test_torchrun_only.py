"""Test: torchrun + from_pretrained, NO deepspeed/zero.init at all."""
import os
import torch

def main():
    print(f"rank={os.environ.get('RANK','?')} torch={torch.__version__}", flush=True)

    # plain torch.distributed init
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print(f"after init_pg: world={world} rank={rank}", flush=True)
    torch.cuda.set_device(rank)

    from transformers import AutoModelForCausalLM
    print(f"rank={rank}: loading Qwen2.5-1.5B-Instruct (sdpa, on cuda:{rank})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(f"cuda:{rank}")
    print(f"rank={rank}: loaded {type(model).__name__}, params on cuda:{rank}", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
