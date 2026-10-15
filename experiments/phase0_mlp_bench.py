#!/usr/bin/env python3
"""Phase-0 Ascend dense SwiGLU FFN microbenchmark. NOT a serving result."""
import argparse
import datetime
import json
import math
import os
import platform
import statistics
import traceback

import torch
import torch_npu


def bench_events(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts, ends = [], []
    for _ in range(iters):
        a = torch.npu.Event(enable_timing=True)
        b = torch.npu.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        starts.append(a)
        ends.append(b)
    torch.npu.synchronize()
    us = [float(a.elapsed_time(b)) * 1000.0 for a, b in zip(starts, ends)]
    return {"median_us": statistics.median(us),
            "min_us": min(us),
            "max_us": max(us),
            "samples_us": us}


def check(ref, out, atol, rtol):
    a, b = ref.float(), out.float()
    err = (a - b).abs()
    threshold = atol + rtol * a.abs()
    bad = err > threshold
    return {"finite": bool(torch.isfinite(b).all().item()),
            "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            "mismatch": int(bad.sum().item()),
            "max_abs_err": float(err.max().item()),
            "mean_abs_err": float(err.mean().item())}


def run_case(m, h, i, dtype, warmup, iters, atol, rtol):
    x = torch.randn((m, h), device="npu", dtype=dtype)
    w1 = torch.randn((h, 2 * i), device="npu", dtype=dtype) / math.sqrt(h)
    w2 = torch.randn((i, h), device="npu", dtype=dtype) / math.sqrt(i)

    def unfused():
        t = x @ w1
        g, u = t.chunk(2, dim=-1)
        return (torch.nn.functional.silu(g) * u) @ w2

    def native_activation():
        return torch_npu.npu_swiglu(x @ w1) @ w2

    def native_ffn():
        return torch_npu.npu_ffn(x, w1, w2, "swiglu", inner_precise=1)

    variants = {"manual_swiglu": unfused, "native_swiglu": native_activation}
    if dtype == torch.float16 and hasattr(torch_npu, "npu_ffn"):
        variants["native_ffn"] = native_ffn

    row = {"m": m, "h": h, "i": i, "dtype": str(dtype), "variants": {}}
    try:
        ref = unfused().detach()
        torch.npu.synchronize()
    except Exception as exc:
        row["status"] = "REFERENCE_FAILED"
        row["error"] = repr(exc)
        return row

    for name, fn in variants.items():
        try:
            out = fn().detach()
            torch.npu.synchronize()
            accuracy = check(ref, out, atol, rtol)
            if not accuracy["finite"] or not accuracy["allclose"]:
                row["variants"][name] = {"status": "NUMERIC_GATE_FAILED",
                                         "correctness": accuracy}
                continue
            perf = bench_events(fn, warmup, iters)
            row["variants"][name] = {"status": "OK",
                                     "correctness": accuracy,
                                     "event_timing": perf}
        except Exception as exc:
            row["variants"][name] = {"status": "UNSUPPORTED_OR_FAILED",
                                     "error": repr(exc),
                                     "traceback": traceback.format_exc(limit=4)}
    if "native_ffn" not in variants:
        row["variants"]["native_ffn"] = {"status": "NOT_TESTED",
                                          "reason": "documented dense swiglu FFN requires FP16 high-performance, and API must exist"}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", default="1,8,32,128,512",
                   help="Actual token batch dimensions; not request concurrency")
    p.add_argument("--h", type=int, default=5120)
    p.add_argument("--i", type=int, default=13824)
    p.add_argument("--dtypes", default="fp16,bf16")
    p.add_argument("--warmup", type=int, default=12)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--atol", type=float, default=0.05)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--out", default="phase0_mlp_results.json")
    args = p.parse_args()
    if args.iters < 3 or args.warmup < 1:
        p.error("--iters >= 3 and --warmup >= 1 are required")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtypes = [dtype_map[x.strip()] for x in args.dtypes.split(",")]
    ms = [int(x.strip()) for x in args.m.split(",")]
    torch.manual_seed(17)
    report = {
        "schema": "phase0-microbench-v1",
        "status": "MEASURED_LOCAL_ONLY",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "ascend_home": os.getenv("ASCEND_HOME_PATH", "unset"),
        },
        "device": str(torch.npu.get_device_name(0)),
        "params": vars(args),
        "note": "Event timings in eager microbenchmark only. Not equivalent to native vLLM/ACLGraph service.",
        "cases": [],
    }
    for dtype in dtypes:
        for m in ms:
            print(f"case m={m} dtype={dtype}", flush=True)
            report["cases"].append(run_case(
                m, args.h, args.i, dtype, args.warmup, args.iters, args.atol, args.rtol))
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
