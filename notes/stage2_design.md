# Stage 2 多教师 OPD 路由：设计提案（待用户审查）

> 写于 2026-05-02，供用户回来后审查的设计草案。**本文档不代表已实现的代码**，只是 Stage 2 启动前需要锁定的设计选择。

## 目标

把 OPSD 单教师 OPD（teacher 经 `+actor_rollout_ref.ref.model.path` 注入）扩展为多教师 OPD：每条训练样本按 `data_source`（或显式 domain 字段）路由到对应 domain 的教师，所有教师 logits 在 phase 1 内分别 fwd+缓存到 CPU、phase 2 学生统一 backward。

## 现状回顾

`src/opd/opd_worker.py` 的 `update_opd` 是清晰的两阶段：

- **Phase 1**：`self.ref_module_fsdp` 加载到 GPU → 全量 micro_batch fwd → logits offload 到 CPU → ref_module_fsdp offload 回 CPU。
- **Phase 2**：`self.actor_module_fsdp` 加载到 GPU → fwd + 与缓存的 teacher logits 算 KL/JSD → backward。

`ref_module_fsdp` 是 verl 内部的单 ref slot（来自 `actor_rollout_ref.ref.model.path`）。verl 设计上没有「多 ref」概念。

## 三种实现路径（按 surgery 大小排序）

### 方案 A：N 个 OPDWorker × resource pool 切片（最干净，开销大）

每个教师独立一个 worker 池（K 卡），各持自己 FSDP 的 ref。Driver 拿到 batch 后把每个 sample dispatch 到对应教师 worker pool 跑 phase 1，然后把结果合并送 actor_pool 跑 phase 2。

- ✅ verl 抽象友好，不动 worker 内部
- ❌ K 张卡固定切给每个教师，N=5 教师在 8 卡机上几乎无法 schedule
- ❌ 跨 worker pool 通信复杂

### 方案 B：单 OPDWorker 持 N 个 ref_module_fsdp（推荐）

在 `OPDWorker.__init__` super 完之后**手动构建 N-1 个额外 ref_module_fsdp**（共享 fsdp_config），存进 `self.extra_ref_modules: dict[str, FSDP]`。`update_opd` Phase 1 改成按 `data_source` 分桶遍历教师。

- ✅ 单 worker 池，所有 GPU 用满
- ✅ 复用现有 FSDP+CPU offload 代码
- ⚠️ 需要手写 ref_module 构造（参考 verl `_build_ref_model` 或 `actor_rollout_ref/ref/dp_ref.py`）
- ⚠️ N 个 ref FSDP 在 CPU 都要 ~7B FP16 ≈ 14 GB → 5 教师 70 GB CPU mem/rank（机器有 1.7 TB，OK）

### 方案 C：单 ref_module_fsdp 复用、按 batch 切换 weights（最 hacky）

每次 phase 1 进入时，根据当前 batch 的 majority data_source 加载对应教师的 weights 到 ref_module_fsdp（其它教师 weights 在 CPU）。一个 batch 内只能用一个教师。

- ✅ 改动最小（基本只改 init + 一个 weight loader）
- ❌ 一个 batch 必须同 domain → 浪费 batch 多样性，data shuffling 后会频繁切换 weights，IO 瓶颈
- ❌ 与 GRPO-on-rollout 思路（混合 domain）不兼容

## 推荐：方案 B

理由：

1. 8×H100 上单 worker pool 是标准 verl 玩法
2. CPU offload 后 N 个 ref 的内存压力可控（70 GB / 5 ref vs 1.7 TB RAM）
3. Phase 1 内部按 data_source 分桶，每个教师只 forward 自己的样本子集 → 计算量与单教师 OPD 相当（不是 N 倍）
4. Phase 2 完全不变（学生只关心缓存的 teacher logits + student logits 形状对齐）

## 关键改动清单（方案 B）

### 1. Config schema (`src/opd/config/opd_trainer.yaml`)

新增 `opd.teachers` 列表（取代单 `+actor_rollout_ref.ref.model.path`）：

```yaml
opd:
  teachers:
    - name: math
      path: /home/ubuntu/models/qwen2.5/teacher-math-yukang
      data_sources: [math_dapo, math, aime24, aime25]
    - name: medical
      path: /home/ubuntu/models/qwen2.5/teacher-medical-umls
      data_sources: [medical_mcq]
  default_teacher: math   # 兜底，未匹配的 data_source 用它
```

向后兼容：若 `opd.teachers` 缺省，回退到 verl 原 `actor_rollout_ref.ref.model.path` 的单教师行为。

### 2. OPDWorker init (`src/opd/opd_worker.py`)

```python
def __init__(self, config, role, **kwargs):
    if role == "actor_rollout":
        role = "actor_rollout_ref"
    super().__init__(config=config, role=role, **kwargs)

    teachers_cfg = config.get("opd", {}).get("teachers")
    if teachers_cfg:
        # First teacher already built by super() into self.ref_module_fsdp via
        # actor_rollout_ref.ref.model.path. Verify it matches teachers_cfg[0].path.
        # Build extra refs:
        from verl.utils.fsdp_utils import build_fsdp_ref  # exact symbol TBD
        self.extra_refs: dict[str, FSDP] = {}
        for t in teachers_cfg[1:]:
            self.extra_refs[t["name"]] = build_fsdp_ref(t["path"], config.actor_rollout_ref.ref.fsdp_config)
        self._teacher_name_for_data_source = {
            ds: t["name"] for t in teachers_cfg for ds in t["data_sources"]
        }
        self._teacher_modules = {teachers_cfg[0]["name"]: self.ref_module_fsdp, **self.extra_refs}
        self._default_teacher = config.opd.get("default_teacher", teachers_cfg[0]["name"])
```

### 3. Phase 1 routing (`update_opd`)

```python
# Group micro_batches by teacher
from collections import defaultdict
buckets = defaultdict(list)  # teacher_name -> list of (mb_index, micro_batch)
for i, mb in enumerate(micro_batches):
    ds_arr = mb.non_tensor_batch.get("data_source", [self._default_teacher] * len(mb))
    # Single teacher per micro_batch (require batches to be sorted by data_source upstream).
    teacher_name = self._teacher_name_for_data_source.get(ds_arr[0], self._default_teacher)
    buckets[teacher_name].append((i, mb))

teacher_logits_cache = [None] * len(micro_batches)
for teacher_name, items in buckets.items():
    ref_mod = self._teacher_modules[teacher_name]
    if ref_needs_offload:
        load_fsdp_model_to_gpu(ref_mod)
    ref_mod.eval()
    for mb_idx, mb in items:
        ...  # existing forward path, but use ref_mod
        teacher_logits_cache[mb_idx] = (logits.to("cpu"), True)
    if ref_needs_offload:
        offload_fsdp_model_to_cpu(ref_mod)
    torch.cuda.empty_cache()
```

### 4. Data 流上游：按 data_source 排序 micro_batch

OPSD 现在的 collate 是随机的 → 一个 micro_batch 内可能混 data_source。最简单：在 `_pad_opd_batch_for_dispatch` 之前按 `data_source` sort，保证每个 micro_batch 同 source。

## 验证矩阵

| 测试 | 方法 |
|---|---|
| 形状/路由正确 | 用 mock 教师（identity head），构造已知 data_source 的 batch，断言路由到正确 teacher |
| 等价性回归 | `opd.teachers=[math_only]` 与原单教师 OPD 在同一 batch 上 loss 应当 byte-for-byte 等同 |
| 多教师不退化 | math + medical，2 epoch DAPO+MedQA，math eval 不应低于单教师 baseline >2pt |

## 开放问题（需要用户决定）

1. **第一次 Stage 2 跑用哪两个教师？** Blueprint 暗示 math+medical 最简，但搜索/工具 reward function 还没 ready，medical reward 也需要写（MedQA MCQ 解析）。考虑只跑 **math + math**（同 domain 双教师，Yukang + FutureMa 备选），先验证路由代码正确，再扩 domain？
2. **FSDP ref 如何构造？** verl 0.7.1 的 `_build_ref_model` 内部可见 API 是否稳定？需要花半天读 verl `workers/fsdp_workers.py` 源码或考虑 monkey-patch。
3. **如果 N 教师 CPU offload 后 RAM 仍紧张？** 7B FP16 × 5 = 70 GB / rank × 8 ranks = 560 GB，加 vLLM/student/optimizer，1.7 TB 应足；但要监控。
4. **Stage 2 batch 是否仍混合 data_source？** 路由切片 vs 同质 micro_batch 是设计抉择。同质实现简单，混合需更细的 per-row 切片（teacher_logits cache 和 dispatch 都要变）。

## 推荐启动顺序

1. **B0 + B2 完成后**（提供单教师 anchor 数）→ 锁定 baseline
2. **方案 B 实现**：约 200-300 LoC（worker + config + 排序 + 单测）
3. **回归测试**：单教师等价 + 双教师 mock 路由
4. **Iter 1 跑 math+math 验证正确性**（Yukang 主，FutureMa 备）
5. **Iter 2 引入第 2 个 domain（medical SFT）**：写 MedQA MCQ reward
6. **Iter 3+：search / tool / code**：每个新 domain 至少 1 周（reward + 数据）
