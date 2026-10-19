# Phase 0 — Discovery contract (2026-10-14)

Status: PLANNED / SOURCE-VERIFIED / NOT HARDWARE-MEASURED. No new experimental results are claimed here.

## Premise and the essential negative control

Ascend OpPlugin already exposes torch_npu.npu_ffn with dense swiglu activation on Atlas A2, **FP16 high-performance only for swiglu**, N1=2*K2, and support for graph mode according to its published API documentation. This is a *mandatory* strong baseline, not a hypothetical new implementation. The same API documentation warns that FFN fusion may regress performance; it suggests assessing whether vector ops >=30us contribute >10% of network operator time. This documentation-level heuristic is **not a locally demonstrated threshold**.

- Vendor API: https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_ffn.md
- vLLM model: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py
- Ascend activation: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/activation.py
- Graph-mode documentation (current; NOT proof of old v0.23 configuration): https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/graph_mode.html
- Qwen config: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json

Qwen2MLP currently expresses merged gate/up -> activation -> down as three distinct layer-level calls, but this Python code alone does **not** prove three hardware launches, DRAM materialization, or whether a graph/compiler merges them. Map the installed vLLM+Ascend code and actual profiler trace before labeling runtime behavior.

## 1. Research question *family*, not a prematurely accepted problem

Does dense MLP execution on Ascend 910B2 leave significant and removable end-to-end serving latency due to:
- P1: gate/up or post-SwiGLU **HBM materialization** between Cube/Vector steps?
- P2: **Cube/Vector scheduling/producer-consumer waits**, even when HBM traffic is not dominant?
- P3: **Host launch/graph boundaries**, shape bucketing or dynamic token batch size preventing fused execution?
- P4: existing native fused FFN's **dtype/shape/layout/precision restrictions** making it unusable or slower than baseline?

No P1–P4 is asserted as true. If no residual remains after native FFN/graph/runtime baselines, stop paper work; small upstream improvements may still be worthwhile.

## 2. Exact dense SwiGLU dataflow, initial shape, and bounds

Define X[M,H], W_gu[H,2I], W_d[I,H], T=X W_gu [M,2I], Z=SiLU(T[:,:I])*T[:,I:] [M,I], Y=Z W_d [M,H]. For H=5120 and I=13824 (Qwen2.5-14B-Instruct config), TP=1, no bias:

- GEMM FLOPs per layer, ignoring activation: (4MHI+2MHI)=6MHI.
- Weight payload per layer for FP16 = (2HI+IH)*2=6HI bytes = 424,673,280 B (~405 MiB), **not** a prediction of per-step DRAM bytes.
- If *both* intermediates are fully written to and reread from HBM, avoidable tensor traffic ceiling is 2*(2MI+MI)*s=6MIs bytes; s=2 for FP16. At M=1: 165,888 B; at M=128: 21,233,664 B (~20.25 MiB).
- This is an **optimistic, conditional upper bound on nominal intermediate tensor traffic**, not measured or necessarily removable HBM traffic. Official Ascend 220x/A2 architecture documentation states that AIC–AIV exchange occurs via Global Memory (GM), so full elimination may be physically impossible when vector work stays on AIV. GM addressing does not prove off-chip HBM traffic, because caches may intervene. Internal buffering, extra transfers, recomputation, partial fusion and layouts change actual bytes. Source: https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html
- If critical-path fraction f is perfectly removable, Amdahl speedup <=1/(1-f), *assuming fixed amount of work and additive critical path*. Sum of overlapping kernel durations is not f. Serving goodput needs direct measurement.
- Ratio of nominal two-intermediate traffic to weight payload is (6MIs)/(3HIs)=2M/H, equal to 0.039% at M=1, 5% at M=128, and 20% at M=512. The crossover is M=H/2=2560 if weights are moved exactly once and intermediates are fully round-tripped; this is **volume accounting only**, not real HBM bandwidth or latency. Small-M decode likely needs investigation of launch/waits/low arithmetic intensity, not a presupposed bandwidth claim.

Important dtype asymmetry: official Qwen model config advertises BF16. The historical serving setup mentioned FP16. **Never substitute an FP16-only fused operator into an existing BF16 model and report an equivalent algorithm.** Benchmark both modes; label an unsupported BF16 fusion as unsupported, not slower.

## 3. Strong baselines and controlled interventions

L0: three dense MLP implementations on identical X/W and dtype:
- A: one merged gate/up MatMul, explicit SiLU*up, down MatMul (simple model-level reference).
- B: merged gate/up MatMul, official torch_npu.npu_swiglu, down MatMul (current NPU activation baseline).
- C: official torch_npu.npu_ffn(..., "swiglu", inner_precise=1) only where supported (FP16 dense FFN).
Separate split gate/up GEMMs may be investigated as an *inferior diagnostic* baseline but **never** the only comparator.

L1: native vLLM-Ascend service baseline on Qwen2.5-14B-Instruct TP1, with exact engine config and actual model dtype. Independent matched measurements for decode-heavy, prefill-heavy, mixed traffic; C1/C4/C8/C16 are request concurrency, **not** equivalent to M in L0. Identify Graph mode, actual batch sizes, sampling, cache and preemptions.

L2: short profiler samples from L1. Align stage, Host ACL timeline, op-level AIC/AIV and MTE, wait/transfer where counters exist, and graph replay. Record whether gate-up/swiglu/down execute as separate NPU tasks and whether there are visible DRAM intermediates. Operator names and tensor shapes must come from actual traces.

L3 (conditional): minimal source-level ablation swapping to the documented C baseline where semantic, dtype/layout and graph conditions are met; then independent unprofiled A/B on the native serving path. Do not implement a bespoke fused kernel before L0–L2.

## 4. Experimental matrix / receipts

| Axis | Initial values | Confounder |
|---|---|---|
| Shape M (isolated) | 1,8,32,128,512 | Does not equal request concurrency |
| Dense dimensions | H=5120,I=13824 | TP1 only; read actual model config |
| Dtype | FP16,BF16 | C only documented for FP16 swiglu |
| Execution | eager first; graph second if supported | compiler fuse/capture makes comparisons different |
| Serving load | decode, prefill, mixed | KV/attention/scheduler costs |
| Device | 1 visible 910B2; re-identify | historical environment is not present evidence |
| Versions | CANN, torch_npu, vLLM, vLLM-Ascend SHA | public main differs from installed v0.23 |

Minimum data per variant: success/exception, output error, NPU event latency distribution, warmup/iterations, shape, dtype, source SHA, device identity. Separate output correctness gate from speed. Keep raw traces out of Git; commit small summaries and checksums only. Favor ABBA order / three independent runs once smoke passes. For profile sample size choose a short case with validated unprofiled behavior, do not compare profiler-on timings to profiler-off throughput.

Operator profiling tool reference (support is version-dependent): https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/profiling.md

## 5. First falsifiers / decision

- If C outperforms B substantially *only* in microbenchmark but actual vLLM is already graph-fused, end-to-end speed may be nil.
- If BF16 is the production path and the API only supports FP16 swiglu, changing model dtype is not a zero-cost ablation; evaluate a BF16-compatible alternative or stop.
- If removable MLP critical-path fraction is small after strongest official baseline, NO-GO for whole-model paper direction.
- If P1 HBM traffic is tiny but P2 Cube/Vector producer waits remain long, reframe toward scheduling (subject to existing work overlap).
- If the best result is a fixed-shape specialized kernel without workload-general residual, classify ENGINEERING / PR, not an established research novelty.

## 6. Commands (run in verified Ascend environment)

~~~bash
npu-smi info
python -c 'import torch,torch_npu; print(torch.__version__,torch_npu.__version__,torch.npu.get_device_name(0))'
python experiments/phase0_mlp_bench.py --m 1,8,32 --dtypes fp16,bf16 --warmup 12 --iters 40 --out phase0_smoke.json
# After passing the smoke/numeric gates:
python experiments/phase0_mlp_bench.py --m 1,8,32,128,512 --dtypes fp16,bf16 --warmup 20 --iters 80 --out phase0_full.json
~~~

DO NOT interpret this script as service profiling. Inspect exceptions/NUMERIC_GATE_FAILED and clock drift; event timing availability varies with torch_npu version. Next action is to use the known-good service harness and the pre-existing Ascend-Inference-Profile-Map profiling runbook.
