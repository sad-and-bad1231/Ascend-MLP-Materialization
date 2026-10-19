# Ascend MLP Materialization

**Stage:** Phase 0 / Bottleneck discovery, not a committed paper mechanism.

We investigate *whether* and *where* dense SwiGLU MLP execution leaves removable NPU/serving overhead, rather than assuming intermediate materialization is the dominant cost.

- [Discovery contract, cost model and experiments](research/PHASE0_DISCOVERY.md)
- [Related work, strong industrial baselines and overlap hazards](research/RELATED_WORK_MAP.md)
- [Phase-0 FP16/BF16, native-vs-separate operator microbenchmark](experiments/phase0_mlp_bench.py)

**Hard gate:** Ascend's existing torch_npu.npu_ffn supports an FP16 high-performance dense swiglu path on the Atlas A2 family according to vendor API documentation. It is the comparator, **not** the proposed research contribution. Qwen2.5-14B-Instruct config advertises BF16, so dtype parity must be preserved when measuring serving.

No hardware tests have been run as part of this repository setup. Never label a profiler-on isolated op speedup as an end-to-end service speedup.

Reference learning/evidence repositories: private Ascend-Inference-Profile-Map and kvrc; public awesome-LLM-inference-systems.
