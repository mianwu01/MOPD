import os
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'

def main():
    print("=== test 1: pure torch + flash_attn ===", flush=True)
    import torch, flash_attn
    print(f"torch {torch.__version__}, flash {flash_attn.__version__}", flush=True)
    x = torch.randn(2, 8, 16, 64, device='cuda', dtype=torch.bfloat16)
    out = flash_attn.flash_attn_func(x, x, x)
    print(f"flash_attn fwd OK: {out.shape}", flush=True)

    print("\n=== test 2: vllm offline LLM small load ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("vllm load OK", flush=True)
    out = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=8))
    print(f"vllm gen OK: {out[0].outputs[0].text!r}", flush=True)


if __name__ == "__main__":
    main()
