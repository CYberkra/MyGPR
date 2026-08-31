#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPR input format registry used by import dialogs and loader routing.

The registry is intentionally descriptive.  Native readers may support only a
safe subset of a vendor format; unsupported variants must fail with a clear
message rather than silently mis-reading binary data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPRFormatSpec:
    key: str
    display_name: str
    extensions: tuple[str, ...]
    sidecars: tuple[str, ...] = ()
    support: str = "native"  # native / recognized / optional / external
    notes: str = ""

    def matches(self, suffix: str) -> bool:
        return suffix.lower().lstrip(".") in {e.lstrip(".").lower() for e in self.extensions}


GPR_FORMAT_SPECS: tuple[GPRFormatSpec, ...] = (
    GPRFormatSpec(
        key="mygpr_csv",
        display_name="MyGPR / matrix CSV",
        extensions=("csv", "txt"),
        support="native",
        notes="二维矩阵 CSV/TXT 或 MyGPR UAV stacked CSV。",
    ),
    GPRFormatSpec(
        key="gprmax_out",
        display_name="gprMax output",
        extensions=("out",),
        sidecars=("in",),
        support="native",
        notes="gprMax HDF5 .out；同目录 .in 用于补充道间距。",
    ),
    GPRFormatSpec(
        key="numpy_array",
        display_name="NumPy array",
        extensions=("npy", "npz"),
        support="native",
        notes="内部/研究交换格式；必须是二维 B-scan 矩阵。",
    ),
    GPRFormatSpec(
        key="mala_rd",
        display_name="MALÅ RD3/RD7 + RAD",
        extensions=("rd3", "rd7", "rad"),
        sidecars=("rad", "rd3", "rd7"),
        support="native-subset",
        notes="读取 signed-int sequential trace 子集；需同名 .rad 头文件。",
    ),
    GPRFormatSpec(
        key="impulseradar_iprb",
        display_name="ImpulseRadar IPRB + IPRH",
        extensions=("iprb", "iprh"),
        sidecars=("iprh", "iprb"),
        support="native-subset",
        notes="读取 CrossOver style 16/32-bit sequential trace 子集；需 .iprh。",
    ),
    GPRFormatSpec(
        key="segy",
        display_name="SEG-Y fixed-length profile",
        extensions=("sgy", "segy"),
        support="native-subset",
        notes="支持固定采样数、常见 int16/int32/float32 big-endian profile。复杂 Rev2 扩展建议外部转换。",
    ),
    GPRFormatSpec(
        key="envi_bsq",
        display_name="ENVI BSQ DAT/HDR",
        extensions=("dat", "hdr"),
        sidecars=("hdr", "dat"),
        support="native-subset",
        notes="支持 ENVI band-sequential 二维/三维数组的轻量导入。",
    ),
    GPRFormatSpec(
        key="sensors_software_dt1",
        display_name="Sensors & Software DT1/HD",
        extensions=("dt1", "hd"),
        sidecars=("hd", "dt1"),
        support="native-subset",
        notes="读取 int16 逐道数据（128B 道头）；需同名 .hd 文本头，ft 单位自动换算为 m。",
    ),
    GPRFormatSpec(
        key="gssi_dzt",
        display_name="GSSI DZT",
        extensions=("dzt", "dzg"),
        sidecars=("dzg",),
        support="native-subset",
        notes="解码单通道 uint8/uint16/int32 profile；DZX/DZG 辅助文件暂不解析。",
    ),
    GPRFormatSpec(
        key="oko_gpr",
        display_name="Geotech OKO GPR/GPR2",
        extensions=("gpr", "gpr2"),
        support="native-subset",
        notes="解码 OKO-2 .GPR2（RGPR readGPR2 布局：444B 头 + 均衡数组 + 36B 道头 float32）；.GPR 旧版格式同布局尝试解码，魔数不符则报错。",
    ),
)


_EXTENSION_TO_SPEC = {
    ext.lower().lstrip("."): spec for spec in GPR_FORMAT_SPECS for ext in spec.extensions
}


def normalize_extension(path_or_suffix: str | Path) -> str:
    text = str(path_or_suffix)
    suffix = Path(text).suffix if not text.startswith(".") else text
    return suffix.lower().lstrip(".")


def get_format_spec(path_or_suffix: str | Path) -> GPRFormatSpec | None:
    return _EXTENSION_TO_SPEC.get(normalize_extension(path_or_suffix))


def is_known_gpr_format(path_or_suffix: str | Path) -> bool:
    return get_format_spec(path_or_suffix) is not None


def supported_file_dialog_filter() -> str:
    common_exts = sorted({f"*.{ext}" for spec in GPR_FORMAT_SPECS for ext in spec.extensions})
    native_exts = sorted(
        {
            f"*.{ext}"
            for spec in GPR_FORMAT_SPECS
            if spec.support.startswith("native")
            for ext in spec.extensions
        }
    )
    recognized_exts = sorted(
        {
            f"*.{ext}"
            for spec in GPR_FORMAT_SPECS
            if spec.support == "recognized"
            for ext in spec.extensions
        }
    )
    return ";;".join(
        [
            "GPR 数据文件 (" + " ".join(common_exts) + ")",
            "当前可直接读取 (" + " ".join(native_exts) + ")",
            "已识别需转换 (" + " ".join(recognized_exts) + ")",
            "所有文件 (*)",
        ]
    )


def compatibility_table() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in GPR_FORMAT_SPECS:
        rows.append(
            {
                "format": spec.display_name,
                "extensions": ", ".join("." + ext for ext in spec.extensions),
                "sidecars": ", ".join("." + ext for ext in spec.sidecars) or "--",
                "support": spec.support,
                "notes": spec.notes,
            }
        )
    return rows


__all__ = [
    "GPRFormatSpec",
    "GPR_FORMAT_SPECS",
    "get_format_spec",
    "is_known_gpr_format",
    "supported_file_dialog_filter",
    "compatibility_table",
]
