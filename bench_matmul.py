#!/usr/bin/env python
"""Peak-FLOPS matrix multiply (GEMM) benchmark for the Intel XPU device.

Measures achieved TFLOPS for square matmul of size N x N across dtypes:
fp64, fp32, fp16, bf16, int8.
"""

import argparse
import time

import torch


def bench_matmul(dtype, n, iters, warmup=5, label=""):
    if dtype == torch.int8:
        a = torch.randint(-128, 127, (n, n), dtype=torch.int8, device="xpu")
        b = torch.randint(-128, 127, (n, n), dtype=torch.int8, device="xpu")
    else:
        a = torch.randn(n, n, dtype=dtype, device="xpu")
        b = torch.randn(n, n, dtype=dtype, device="xpu")

    for _ in range(warmup):
        a @ b
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.xpu.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    flops = 2.0 * n**3
    tflops = flops / elapsed / 1e12
    print(f"{label:>10s} N={n:<6d} {elapsed*1e3:8.2f} ms  {tflops:8.2f} TFLOPS")
    return tflops


def main():
    p = argparse.ArgumentParser(description="XPU matmul peak-FLOPS benchmark")
    p.add_argument("--n", type=int, default=8192, help="matrix size N (square NxN)")
    p.add_argument("--iters", type=int, default=10, help="measured iterations")
    p.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    args = p.parse_args()

    print(f"device      : {torch.xpu.get_device_name(0)}")
    props = torch.xpu.get_device_properties(0)
    print(f"EUs         : {props.gpu_eu_count}")
    print(f"subslices   : {props.gpu_subslice_count}")
    print(f"memory      : {props.total_memory/2**30:.1f} GiB")
    print(f"N           : {args.n}  iters: {args.iters}  warmup: {args.warmup}")
    print("-" * 56)

    bench_matmul(torch.float64, args.n, args.iters, args.warmup, "fp64")
    bench_matmul(torch.float32, args.n, args.iters, args.warmup, "fp32")
    bench_matmul(torch.float16, args.n, args.iters, args.warmup, "fp16")
    bench_matmul(torch.bfloat16, args.n, args.iters, args.warmup, "bf16")
    bench_matmul(torch.int8, args.n, args.iters, args.warmup, "int8")


if __name__ == "__main__":
    main()
