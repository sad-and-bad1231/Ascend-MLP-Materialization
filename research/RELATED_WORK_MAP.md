# Related Work & industrial baselines — preliminary audited map, 2026-10-14

**Interpretation policy:** the following problem/mechanism/limit fields are grounded in published abstracts, official software docs and selected source code. They are not full-paper experiment replication. “Limit for our purpose” means scope separation, **not** an unproved deficiency of the paper.

| Priority | Work (provenance) | The problem *they* solve | Identifiable method / what we should learn | NOT evidence of |
|---|---|---|---|---|
| P0 | Ascend OpPlugin npu_ffn (official operator API) | FFN/MoE FFN execution and optional fusion on Atlas A2/A3 | Dense FP16 swiglu FFN with two MatMuls and activation under published dimension/precision restrictions; warning that fusion can regress. **Mandatory industrial comparator** | Actual data traffic or our vLLM graph launch pattern; a free BF16 replacement |
| P0 | XY-Serve (ASPLOS 2026, Huawei/Tsinghua, DOI 10.1145/3760250.3762228) | Dynamic LLM serving shapes poorly map to fixed efficient tiled accelerator kernels | Meta-kernel abstractions, dynamic-shape virtual padding for GEMM, hybrid serving scheduling; Ascend end-to-end context | Claim of having specifically solved *all* dense SwiGLU interstage materialization |
| P0 | MPK (OSDI 2026, CMU etc.) | Per-operator GPU scheduling blocks cross-operator pipelining/fine-grained overlap | SM-level dependency graph, persistent mega-kernel compiler and in-kernel scheduling | Direct portable implementation or measured 910B2 benefits |
| P1 | Event Tensor (MLSys 2026) | Launch gaps and synchronization; dynamic shape/data-dependent megakernels hard to compile | Tiled task dependencies as Event Tensor, static/dynamic scheduling | Ascend native FFN implementation or simple one-op fusion recipe |
| P1 | NanoFlow (OSDI 2025, UW etc.) | Serial execution underutilizes heterogeneous compute/memory/network | Nano-batching, resource-aware intra-device overlap, optimization over nano-batch allocation | A proof that MLP tensor intermediates dominate the critical path |
| P1 | DynaFlow (MLSys 2026, UW etc.) | Intra-device operator overlap is invasive and context-sensitive | Separates logical dataflow and physical scheduling, asynchronous backend, CUDA Graph compatibility | Ascend Cube/Vector workload attribution |
| P1 | Mirage (OSDI 2025, CMU etc.) | Tensor-program optimization needs joint algebra/kernel/block/thread search | Multi-level µGraph and equivalence verification | That novel algebraic fusion is necessary before trying vendor kernel |
| P2 | KPerfIR (OSDI 2025) | Compiler/profile attribution gaps in GPU kernel performance | Connects pipeline utilization and dataflow to compiler IR | Physical 910B2 counter availability |
| P2 | Twill (OSDI 2026) | Software pipelining and warp specialization scheduling on NVIDIA Tensor Core GPUs | Constraint-based joint SWP+WS optimization | Ascend Cube/Vector schedule equivalence |
| P2 | QFactory (USENIX ATC 2025) | Quantization transforms add conversion/reformat overhead in serving kernels | Qtile Graph across dequant/fused GEMM and tile schedule | A dense unquantized MLP fusion mechanism |

## Source-of-truth links

- Ascend API: https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_ffn.md
- vLLM Ascend swiglu: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/activation.py
- vLLM dense Qwen2MLP: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py
- XY-Serve DOI: https://doi.org/10.1145/3760250.3762228 ; ASPLOS program: https://www.asplos-conference.org/asplos2026/program/
- MPK: https://www.usenix.org/conference/osdi26/presentation/cheng
- Event Tensor: https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html
- NanoFlow: https://www.usenix.org/conference/osdi25/presentation/zhu-kan
- DynaFlow: https://proceedings.mlsys.org/paper_files/paper/2026/hash/bbd7d8bd780fcf7143add2317ba04638-Abstract-Conference.html
- Mirage: https://www.usenix.org/conference/osdi25/presentation/wu-mengdi
- KPerfIR: https://www.usenix.org/conference/osdi25/presentation/guan
- Twill: https://www.usenix.org/conference/osdi26/presentation/soi
- QFactory: https://www.usenix.org/conference/atc25/presentation/zhang-qihao
- Personal library: https://github.com/sad-and-bad1231/awesome-LLM-inference-systems/blob/main/topics/kernels-compilers.md
- Lab profiling rules are in the private Ascend-Inference-Profile-Map repository.

## Main novelty hazards, ordered

1. Existing vendor npu_ffn might already remove the purported intermediate traffic; test it before any custom kernel work.
2. Real model BF16 vs native fused swiglu FP16 capability mismatch; do not use changing numerical type as a fair baseline.
3. Current serving graph/compile path may already absorb launch overhead and fuse portions of the graph; trace installed software, not GitHub main.
4. XY-Serve already addresses Ascend-native dynamic shape/tiling at the serving level. A mere "dynamic-shape GEMM optimization" is weak unless a distinct residual is measured.
5. MPK/Event Tensor/DynaFlow cover cross-op scheduling abstractions. A local hard-coded overlap must establish hardware-specific irreducibility and a strong system outcome.

## Reading sequence

1. OpPlugin API + vLLM Qwen2MLP + vLLM-Ascend activation/Graph Mode.
2. XY-Serve for architecture-native dynamic shape and system measurement.
3. MPK and Event Tensor for fusion-vs-scheduling distinction.
4. NanoFlow and DynaFlow for whole-step critical path and overlap.
5. Mirage/KPerfIR/Twill only if profiler finds a schedule/compiler-level residual.

No paper title is a novel-contribution claim. The goal is to rule out existing fixes and localize a **remaining physical execution constraint**.
