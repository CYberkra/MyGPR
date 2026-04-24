#!/usr/bin/env python3
"""测试读取 gprMax 输出文件并显示正确的 B-scan"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


def test_read_gprmax_out(folder_path):
    """测试读取并显示 gprMax 数据"""

    # 找到所有单独的 .out 文件
    out_files = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".out") and "merged" not in f and "fixed" in f
        ],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )

    print(f"找到 {len(out_files)} 个 .out 文件")

    if len(out_files) == 0:
        print("没有找到 .out 文件")
        return

    # 读取第一个文件获取参数
    first_path = os.path.join(folder_path, out_files[0])
    with h5py.File(first_path, "r") as f:
        attrs = dict(f.attrs)
        iterations = int(attrs.get("Iterations", 0))
        dt = float(attrs.get("dt", 0))
        data0 = f["rxs/rx1/Ez"][:]

    samples = len(data0)
    n_traces = len(out_files)

    print(f"参数: iterations={iterations}, dt={dt}")
    print(f"采样点数: {samples}, 道数: {n_traces}")

    # 创建合并矩阵
    matrix = np.zeros((samples, n_traces), dtype=np.float32)
    matrix[:, 0] = data0

    # 读取其他文件
    for i, fname in enumerate(out_files[1:], 1):
        fpath = os.path.join(folder_path, fname)
        with h5py.File(fpath, "r") as f:
            matrix[:, i] = f["rxs/rx1/Ez"][:]

    # 保存合并后的文件
    merged_path = os.path.join(folder_path, "gpr_model_test_merged.out")
    with h5py.File(merged_path, "w") as f:
        f.create_dataset("rxs/rx1/Ez", data=matrix)
        f.attrs["Iterations"] = iterations
        f.attrs["dt"] = dt
        f.attrs["nx_ny_nz"] = [1, 1, 1]
        f.attrs["Title"] = "Test Merged B-scan"

    print(f"已保存合并后的文件: {merged_path}")
    print(f"数据形状: {matrix.shape}")

    # 显示 B-scan
    vmax = np.max(np.abs(matrix)) * 0.30

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 原始数据显示
    im1 = axes[0].imshow(
        matrix,
        cmap="seismic",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
        extent=[0, n_traces, samples * dt * 1e9, 0],
    )
    axes[0].set_title("GPR B-scan (Raw)")
    axes[0].set_xlabel("Trace Number")
    axes[0].set_ylabel("Time (ns)")
    plt.colorbar(im1, ax=axes[0], label="Amplitude")

    # 显示数据范围
    print(f"数据范围: {matrix.min():.6f} 到 {matrix.max():.6f}")
    print(f"数据均值: {matrix.mean():.6f}")
    print(f"数据标准差: {matrix.std():.6f}")

    # 检查是否有明显的信号模式
    max_per_trace = np.max(np.abs(matrix), axis=0)
    strong_traces = np.where(max_per_trace > max_per_trace.mean() * 2)[0]
    print(f"强信号道数: {len(strong_traces)}/{n_traces}")

    if len(strong_traces) > 0:
        print(f"强信号道位置: {strong_traces[:10]}...")

    # AGC 增益显示
    agc_window = 50
    agc_gain = np.zeros_like(matrix)
    for i in range(matrix.shape[1]):
        trace = matrix[:, i]
        envelope = np.abs(trace)
        # 滑动平均
        kernel = np.ones(agc_window) / agc_window
        smoothed = np.convolve(envelope, kernel, mode="same")
        smoothed = np.maximum(smoothed, 1e-10)
        agc_gain[:, i] = trace / smoothed

    vmax_agc = np.max(np.abs(agc_gain)) * 0.30

    im2 = axes[1].imshow(
        agc_gain,
        cmap="seismic",
        aspect="auto",
        vmin=-vmax_agc,
        vmax=vmax_agc,
        origin="upper",
        extent=[0, n_traces, samples * dt * 1e9, 0],
    )
    axes[1].set_title("GPR B-scan (AGC Gain)")
    axes[1].set_xlabel("Trace Number")
    axes[1].set_ylabel("Time (ns)")
    plt.colorbar(im2, ax=axes[1], label="Amplitude")

    plt.tight_layout()

    # 保存图像
    output_png = os.path.join(folder_path, "gpr_model_test_bscan.png")
    plt.savefig(output_png, dpi=150)
    plt.close()

    print(f"B-scan 图像已保存: {output_png}")

    return merged_path


if __name__ == "__main__":
    folder = r"D:\ClawX-Data\sim\gprmax_outcsv\gpr_model_20260403_020147"
    test_read_gprmax_out(folder)
