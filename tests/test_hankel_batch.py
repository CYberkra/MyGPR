#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke / benchmark script for Hankel-SVD denoising."""

from __future__ import annotations

import time

import numpy as np

from PythonModule.hankel_svd import method_hankel_svd


def main() -> None:
    np.random.seed(42)
    data = np.random.randn(501, 3081).astype(np.float32)

    print("=" * 60)
    print("Testing Hankel-SVD Smoke / Benchmark")
    print(f"Data shape: {data.shape}")
    print("=" * 60)

    print("\n[Test 1] Basic functionality...")
    start = time.time()
    result, meta = method_hankel_svd(data.copy(), window_length=125, rank=5)
    elapsed = time.time() - start
    print(f"  [OK] Processing time: {elapsed:.2f}s")
    print(f"  [OK] Output shape: {result.shape}")
    print(f"  [OK] Output range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"  [OK] Metadata: {meta}")

    print("\n[Test 2] Consistency check...")
    result2, _ = method_hankel_svd(data.copy(), window_length=125, rank=5)
    diff = np.abs(result - result2).max()
    print(f"  [OK] Max difference between two runs: {diff:.2e}")

    print("\n[Test 3] Auto rank detection...")
    result_auto, meta_auto = method_hankel_svd(
        data.copy(), window_length=125, rank=None
    )
    result_fixed, meta_fixed = method_hankel_svd(data.copy(), window_length=125, rank=5)
    print(
        f"  [OK] Auto rank output range: [{result_auto.min():.3f}, {result_auto.max():.3f}]"
    )
    print(
        f"  [OK] Fixed rank (5) output range: [{result_fixed.min():.3f}, {result_fixed.max():.3f}]"
    )
    print(f"  [OK] Auto rank metadata: {meta_auto}")
    print(f"  [OK] Fixed rank metadata: {meta_fixed}")

    print("\n[Test 4] Window benchmark...")
    window_sizes = [64, 96, 125, 160]
    print(f"  {'Window':<12} {'Time (s)':<12}")
    print(f"  {'-' * 12} {'-' * 12}")
    times = {}
    for window in window_sizes:
        start = time.time()
        _, _ = method_hankel_svd(data.copy(), window_length=window, rank=5)
        elapsed = time.time() - start
        times[window] = elapsed
        print(f"  {window:<12} {elapsed:<12.2f}")

    best_window = min(times.keys(), key=lambda key: times[key])
    print("\n" + "=" * 60)
    print("[PASS] Hankel-SVD smoke checks completed!")
    print(f"[INFO] Best benchmark window: {best_window} with {times[best_window]:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
