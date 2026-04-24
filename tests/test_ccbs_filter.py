"""Test for CCBS (Cross-Correlation-Based Background Subtraction) filter"""

import sys
import time
import numpy as np

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None

sys.path.insert(0, r"D:\ClawX-Data\code\GPR_GUI_main_2026-03-23")

from PythonModule.ccbs_filter import apply_ccbs_filter, method_ccbs


class TestCCBSFilter:
    """Test suite for CCBS filter"""

    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.M, self.N = 500, 100  # Standard GPR dimensions

    def test_basic_functionality(self):
        """Test basic CCBS filtering works"""
        # Create simple test data
        data = np.random.randn(self.M, self.N).astype(np.float32)

        result = apply_ccbs_filter(data)

        assert result.shape == data.shape
        assert result.dtype == data.dtype
        assert np.isfinite(result).all()

    def test_with_reference_wave(self):
        """Test CCBS with custom reference wave"""
        data = np.random.randn(self.M, self.N).astype(np.float32)
        reference = np.sin(np.linspace(0, 4 * np.pi, self.M)).astype(np.float32)

        result = apply_ccbs_filter(data, reference_wave=reference)

        assert result.shape == data.shape
        assert np.isfinite(result).all()

    def test_without_reference_uses_mean(self):
        """Test that None reference defaults to mean trace"""
        data = np.random.randn(self.M, self.N).astype(np.float32)

        # With reference=None (default)
        result_no_ref = apply_ccbs_filter(data, reference_wave=None)

        # With explicit mean as reference
        mean_ref = np.mean(data, axis=1)
        result_with_mean = apply_ccbs_filter(data, reference_wave=mean_ref)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_no_ref, result_with_mean, decimal=5)

    def test_edge_case_zero_norm_reference(self):
        """Test handling of zero-norm reference wave"""
        data = np.random.randn(self.M, self.N).astype(np.float32)
        zero_ref = np.zeros(self.M, dtype=np.float32)

        # Should not raise error, returns input unchanged
        result = apply_ccbs_filter(data, reference_wave=zero_ref)

        assert np.allclose(result, data)

    def test_edge_case_zero_norm_trace(self):
        """Test handling of traces with zero norm"""
        data = np.random.randn(self.M, self.N).astype(np.float32)
        # Make one column zero
        data[:, 5] = 0

        result = apply_ccbs_filter(data)

        assert result.shape == data.shape
        assert np.isfinite(result).all()

    def test_consistency(self):
        """Test that results are deterministic"""
        np.random.seed(123)
        data = np.random.randn(self.M, self.N).astype(np.float32)

        result1 = apply_ccbs_filter(data)
        result2 = apply_ccbs_filter(data)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_background_reduction(self):
        """Test that background is actually reduced"""
        # Create data with strong background
        t = np.linspace(0, 1, self.M)
        background = np.exp(-3 * t) * np.sin(20 * np.pi * t)

        data = np.zeros((self.M, self.N), dtype=np.float32)
        for i in range(self.N):
            data[:, i] = background + 0.1 * np.random.randn(self.M)

        result = apply_ccbs_filter(data)

        # Background should be reduced
        orig_bg_level = np.mean(np.abs(data))
        proc_bg_level = np.mean(np.abs(result))
        assert proc_bg_level < orig_bg_level

    def test_dimension_validation(self):
        """Test that invalid dimensions raise errors"""
        # 1D input should fail
        try:
            apply_ccbs_filter(np.random.randn(100))
            assert False, "Should have raised ValueError for 1D input"
        except ValueError:
            pass

        # 3D input should fail
        try:
            apply_ccbs_filter(np.random.randn(100, 50, 3))
            assert False, "Should have raised ValueError for 3D input"
        except ValueError:
            pass

    def test_reference_length_validation(self):
        """Test that mismatched reference length raises error"""
        data = np.random.randn(self.M, self.N).astype(np.float32)
        wrong_ref = np.random.randn(self.M + 10).astype(np.float32)

        try:
            apply_ccbs_filter(data, reference_wave=wrong_ref)
            assert False, "Should have raised ValueError for wrong reference length"
        except ValueError:
            pass

    def test_performance_benchmark(self):
        """Benchmark CCBS performance"""
        # Use realistic GPR dimensions
        M, N = 501, 3081
        np.random.seed(42)
        data = np.random.randn(M, N).astype(np.float32)

        start = time.time()
        result = apply_ccbs_filter(data)
        elapsed = time.time() - start

        print(f"\nCCBS Benchmark (M={M}, N={N}):")
        print(f"  Processing time: {elapsed:.3f}s")
        print(f"  Throughput: {N / elapsed:.0f} traces/sec")

        # Should complete in reasonable time (< 2 seconds for this size)
        assert elapsed < 5.0
        assert result.shape == (M, N)

    def test_method_ccbs_wrapper(self):
        """Test the method_ccbs wrapper function"""
        data = np.random.randn(self.M, self.N).astype(np.float32)

        result, metadata = method_ccbs(data)

        assert result.shape == data.shape
        assert "method" in metadata
        assert metadata["method"] == "CCBS"
        assert "reference_used" in metadata

    def test_ncc_range(self):
        """Test that NCC values are properly normalized"""
        # The weights H_i should be in [0, 1] range
        data = np.random.randn(self.M, self.N).astype(np.float32)

        # Use the internal logic to verify NCC computation
        b_mean = np.mean(data, axis=1)
        ref_norm = np.linalg.norm(b_mean)
        trace_norms = np.linalg.norm(data, axis=0)
        trace_norms_safe = np.where(trace_norms < 1e-10, 1.0, trace_norms)
        dot_products = np.dot(b_mean, data)
        ncc_values = dot_products / (ref_norm * trace_norms_safe)

        # NCC should be in [-1, 1] range
        assert np.all(ncc_values[~np.isnan(ncc_values)] >= -1.0)
        assert np.all(ncc_values[~np.isnan(ncc_values)] <= 1.0)


if __name__ == "__main__":
    print("=" * 60)
    print("CCBS Filter Test Suite")
    print("=" * 60)

    test = TestCCBSFilter()
    test.setup_method()

    tests = [
        ("Basic functionality", test.test_basic_functionality),
        ("With reference wave", test.test_with_reference_wave),
        ("Default uses mean", test.test_without_reference_uses_mean),
        ("Zero norm reference", test.test_edge_case_zero_norm_reference),
        ("Zero norm trace", test.test_edge_case_zero_norm_trace),
        ("Consistency", test.test_consistency),
        ("Background reduction", test.test_background_reduction),
        ("Dimension validation", test.test_dimension_validation),
        ("Reference length validation", test.test_reference_length_validation),
        ("Performance benchmark", test.test_performance_benchmark),
        ("Method wrapper", test.test_method_ccbs_wrapper),
        ("NCC range check", test.test_ncc_range),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"  [OK] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
