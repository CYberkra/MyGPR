#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick validation script for 2D MSSA on real GPR data.

Usage:
    python scripts/validate_mssa_on_real_data.py --input path/to/your/data.npy
    python scripts/validate_mssa_on_real_data.py --input path/to/your/data.csv

This will apply 2D MSSA with different aggressiveness levels and save
before/after comparison images.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PythonModule.hankel_svd import method_hankel_svd


def load_data(path: str) -> np.ndarray:
    """Load GPR data from file."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".csv":
        return np.loadtxt(p, delimiter=",")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def plot_comparison(before, after, title, output_path):
    """Plot before/after comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmax = np.percentile(np.abs(before), 99)
    
    axes[0].imshow(before, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Before")
    axes[0].set_xlabel("Trace")
    axes[0].set_ylabel("Sample")
    
    axes[1].imshow(after, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    axes[1].set_title("After")
    axes[1].set_xlabel("Trace")
    
    diff = before - after
    vmax_diff = np.percentile(np.abs(diff), 99)
    axes[2].imshow(diff, aspect="auto", cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title("Difference (Before - After)")
    axes[2].set_xlabel("Trace")
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate 2D MSSA on real GPR data")
    parser.add_argument("--input", required=True, help="Path to input data (.npy or .csv)")
    parser.add_argument("--output-dir", default="output/mssa_validation", help="Output directory")
    parser.add_argument("--window-length", type=int, default=0, help="Window length (0=auto)")
    parser.add_argument("--rank", type=int, default=0, help="Rank (0=auto)")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"Data shape: {data.shape}")
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    # Ensure float64
    data = np.asarray(data, dtype=np.float64)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test different aggressiveness levels
    aggressiveness_levels = [0.3, 0.5, 1.0, 1.5, 2.0]
    
    for agg in aggressiveness_levels:
        print(f"\nProcessing with aggressiveness={agg}...")
        denoised, meta = method_hankel_svd(
            data,
            window_length=args.window_length,
            rank=args.rank,
            aggressiveness=agg,
        )
        
        print(f"  Window length: {meta['window_length']}")
        print(f"  Rank: {meta['effective_rank_max']}")
        print(f"  Rank mode: {meta['rank_selection_mode']}")
        
        # Save comparison image
        plot_comparison(
            data,
            denoised,
            f"2D MSSA (aggressiveness={agg}, rank={meta['effective_rank_max']})",
            output_dir / f"comparison_agg_{agg:.1f}.png",
        )
        
        # Save denoised data
        np.save(output_dir / f"denoised_agg_{agg:.1f}.npy", denoised)
    
    print(f"\nAll results saved to {output_dir}")
    print("\nTips:")
    print("- Lower aggressiveness (0.3-0.5): More conservative, keeps more signal")
    print("- Medium aggressiveness (1.0): Balanced")
    print("- Higher aggressiveness (1.5-2.0): More aggressive denoising")
    print("- If results look too smooth/blurry, try lower aggressiveness or higher rank")
    print("- If noise is still visible, try higher aggressiveness")


if __name__ == "__main__":
    main()
