#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCBS (Cross-Correlation-Based Background Subtraction) filter for GPR data.

This module implements the CCBS algorithm based on recent UAV-GPR signal processing
literature. The method dynamically weights between a reference wave and an average
background trace based on cross-correlation similarity to remove horizontal clutter,
direct-coupling waves, and ground reflections.
"""

import numpy as np
from scipy import signal


def apply_ccbs_filter(b_scan, reference_wave=None):
    """
    Apply Cross-Correlation-Based Background Subtraction (CCBS) filter to GPR B-scan data.

    The CCBS method dynamically weights between a "reference wave" and an "average
    background trace" based on cross-correlation similarity. This adaptive approach
    effectively removes horizontal clutter, direct-coupling waves, and ground
    reflections while preserving target signals.

    Mathematical Model:
    -------------------
    For each trace B_i, the processed output S_i is:

        S_i = B_i - H_i * v - (1 - H_i) * B_mean

    where:
        - B_i: i-th A-scan trace (column i of B-scan)
        - v: reference wave (A-scan with no target)
        - B_mean: global mean trace across all A-scans
        - H_i ∈ [0, 1]: weighting factor based on cross-correlation similarity

    Physical Interpretation of H_i:
    -------------------------------
    H_i represents the similarity between the current trace B_i and the reference
    wave v. It acts as an adaptive mixing parameter:

    - When H_i ≈ 1 (high similarity):
      The trace B_i closely resembles the reference wave v (likely containing
      only background/clutter). The subtraction prioritizes removing v,
      effectively canceling direct-coupling and ground reflections.

    - When H_i ≈ 0 (low similarity):
      The trace B_i differs significantly from v (likely containing targets).
      The subtraction prioritizes removing the mean background B_mean,
      preserving anomalous target signals.

    - Intermediate H_i values (0 < H_i < 1):
      Provide smooth transition between the two subtraction modes,
      handling cases with weak targets or varying background.

    Parameters
    ----------
    b_scan : np.ndarray
        Raw B-scan data array of shape (M, N), where:
        - M: number of time samples (rows, depth dimension)
        - N: number of traces/A-scans (columns, spatial dimension)
    reference_wave : np.ndarray, optional
        1D reference wave array of length M. This represents an A-scan recorded
        when there is no target underground (pure background/clutter).
        If None (default), uses the global mean trace of the B-scan as reference.

    Returns
    -------
    np.ndarray
        Processed B-scan array of shape (M, N) with background removed.
        Same dtype as input b_scan.

    Raises
    ------
    ValueError
        If b_scan is not 2D or if reference_wave length doesn't match b_scan rows.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate B-scan with 500 time samples and 100 traces
    >>> b_scan = np.random.randn(500, 100)
    >>> # Apply CCBS with auto-computed reference (mean trace)
    >>> result = apply_ccbs_filter(b_scan)
    >>> # Apply CCBS with custom reference wave
    >>> reference = np.sin(np.linspace(0, 4*np.pi, 500))
    >>> result = apply_ccbs_filter(b_scan, reference_wave=reference)
    """
    # Validate input dimensions
    if b_scan.ndim != 2:
        raise ValueError(
            f"b_scan must be 2D array with shape (M, N), got {b_scan.ndim}D"
        )

    M, N = b_scan.shape

    # Handle reference wave
    if reference_wave is None:
        # Use global mean trace as default reference
        # This represents the average background across all spatial positions
        reference_wave = np.mean(b_scan, axis=1)
    else:
        reference_wave = np.asarray(reference_wave)
        if reference_wave.shape[0] != M:
            raise ValueError(
                f"reference_wave length ({reference_wave.shape[0]}) must match "
                f"b_scan rows ({M})"
            )

    # Ensure reference is 1D
    reference_wave = reference_wave.reshape(-1)

    # =========================================================================
    # Step 1: Calculate the mean signal (global average trace)
    # =========================================================================
    # B_mean represents the spatial average of all A-scans. It captures the
    # common background structure that is consistent across the survey line,
    # including typical ground reflections and system reverberations.
    b_mean = np.mean(b_scan, axis=1)  # Shape: (M,)

    # =========================================================================
    # Step 2: Calculate Normalized Cross-Correlation (NCC) for all traces
    # =========================================================================
    # We compute the cross-correlation between each trace B_i and reference v.
    # NCC measures the similarity between two signals, normalized by their
    # energies to make it invariant to amplitude scaling.
    #
    # NCC formula: (B_i · v) / (||B_i|| * ||v||)
    # where · denotes dot product and ||·|| denotes L2 norm

    # Pre-compute reference norm (constant for all traces)
    ref_norm = np.linalg.norm(reference_wave)

    # Handle edge case: zero-norm reference wave
    if ref_norm < 1e-10:
        # If reference is effectively zero, no meaningful subtraction possible
        # Return input unchanged to avoid division by zero
        return b_scan.copy()

    # Compute norms for all traces simultaneously (vectorized)
    # axis=0 computes norm along time dimension for each trace
    trace_norms = np.linalg.norm(b_scan, axis=0)  # Shape: (N,)

    # Handle edge case: zero-norm traces (dead channels or zero-padded data)
    # Replace near-zero norms with 1.0 to avoid division by zero
    # Traces with zero norm will naturally have zero correlation
    trace_norms_safe = np.where(trace_norms < 1e-10, 1.0, trace_norms)

    # Compute dot products between reference and all traces (vectorized)
    # Result: correlation coefficients for all N traces at once
    dot_products = np.dot(reference_wave, b_scan)  # Shape: (N,)

    # Calculate Normalized Cross-Correlation coefficients
    # X_i ∈ [-1, 1], where 1 = identical, -1 = opposite, 0 = orthogonal
    ncc_values = dot_products / (ref_norm * trace_norms_safe)

    # Set NCC to 0 for traces that had zero norm (no valid correlation)
    zero_norm_mask = trace_norms < 1e-10
    ncc_values = np.where(zero_norm_mask, 0.0, ncc_values)

    # =========================================================================
    # Step 3: Calculate weighting factors H_i from NCC values
    # =========================================================================
    # H_i maps the correlation coefficient X_i ∈ [-1, 1] to weight ∈ [0, 1].
    #
    # Mapping strategy: Use linear transformation from [-1, 1] to [0, 1]
    # H_i = (X_i + 1) / 2
    #
    # This ensures:
    #   - Perfect correlation (X_i = 1)    → H_i = 1 (use reference wave)
    #   - Zero correlation (X_i = 0)       → H_i = 0.5 (balanced mix)
    #   - Anti-correlation (X_i = -1)      → H_i = 0 (use mean trace)
    #
    # Alternative sigmoid mapping (commented out) provides steeper transition:
    # H_i = 1 / (1 + exp(-k * X_i)) with k=3 for sharper discrimination

    # Linear mapping (default)
    weights = (ncc_values + 1.0) / 2.0  # Shape: (N,)

    # Clip to ensure valid range [0, 1] despite numerical errors
    weights = np.clip(weights, 0.0, 1.0)

    # =========================================================================
    # Step 4: Apply weighted background subtraction (vectorized)
    # =========================================================================
    # Expand dimensions for broadcasting:
    # - b_mean: (M,) -> (M, 1) to subtract from all columns
    # - weights: (N,) -> (1, N) to apply per-trace weighting

    b_mean_2d = b_mean[:, np.newaxis]  # Shape: (M, 1)
    reference_2d = reference_wave[:, np.newaxis]  # Shape: (M, 1)
    weights_2d = weights[np.newaxis, :]  # Shape: (1, N)

    # Compute weighted background for each trace
    # background_i = H_i * v + (1 - H_i) * B_mean
    #              = H_i * v + B_mean - H_i * B_mean
    #              = B_mean + H_i * (v - B_mean)
    #
    # This formulation is numerically stable and clearly shows the interpolation
    # between the two background estimates (mean and reference).

    weighted_background = (
        weights_2d * reference_2d + (1.0 - weights_2d) * b_mean_2d
    )  # Shape: (M, N)

    # Apply subtraction: S = B - background
    processed = b_scan - weighted_background

    return processed


def method_ccbs(data, reference_wave=None, **kwargs):
    """
    Wrapper function for methods_registry compatibility.

    This function wraps apply_ccbs_filter to match the interface expected by
    the methods_registry system for local-type processing methods.

    Parameters
    ----------
    data : np.ndarray
        Input B-scan array of shape (M, N)
    reference_wave : np.ndarray, optional
        Reference wave array. If None, uses mean trace.
    **kwargs : dict
        Additional keyword arguments (ignored, for compatibility)

    Returns
    -------
    tuple
        (processed_data, metadata_dict) where metadata contains algorithm info
    """
    result = apply_ccbs_filter(data, reference_wave=reference_wave)

    metadata = {
        "method": "CCBS",
        "description": "Cross-Correlation-Based Background Subtraction",
        "reference_used": reference_wave is not None,
    }

    return result, metadata


if __name__ == "__main__":
    # Example usage and basic verification
    import time

    print("=" * 60)
    print("CCBS Filter - Example Usage")
    print("=" * 60)

    # Generate synthetic test data
    np.random.seed(42)
    M, N = 500, 200  # 500 time samples, 200 traces

    # Create synthetic B-scan with:
    # 1. Background wave (simulating direct coupling)
    # 2. Random noise
    # 3. Target signals (hyperbolas)

    t = np.linspace(0, 1, M)
    x = np.arange(N)

    # Background: exponential decay + sinusoidal oscillation
    background_wave = np.exp(-3 * t) * np.sin(20 * np.pi * t)

    # Create B-scan
    b_scan = np.zeros((M, N))

    # Add background to all traces (with slight variations)
    for i in range(N):
        variation = 0.9 + 0.2 * np.random.rand()
        b_scan[:, i] = variation * background_wave

    # Add some targets (hyperbolic signatures)
    target_positions = [50, 100, 150]
    for pos in target_positions:
        for i in range(N):
            distance = abs(i - pos)
            if distance < 30:
                depth = 100 + int(5 * np.sqrt(distance))
                if depth < M:
                    b_scan[depth : depth + 5, i] += 2.0

    # Add noise
    b_scan += 0.1 * np.random.randn(M, N)

    print(f"Input shape: {b_scan.shape}")
    print(f"Input range: [{b_scan.min():.3f}, {b_scan.max():.3f}]")

    # Apply CCBS
    start_time = time.time()
    result = apply_ccbs_filter(b_scan)
    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed * 1000:.2f} ms")
    print(f"Output shape: {result.shape}")
    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")

    # Verify background suppression
    bg_reduction = np.mean(np.abs(b_scan)) - np.mean(np.abs(result))
    print(f"\nBackground reduction: {bg_reduction:.3f}")
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
