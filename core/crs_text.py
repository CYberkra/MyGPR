#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize CRS labels stored as either machine IDs or human-readable text."""
from __future__ import annotations

import re
from typing import Any

try:
    from pyproj import CRS
    from pyproj.exceptions import CRSError, ProjError
except ImportError:  # pragma: no cover - pyproj is a required runtime dependency
    CRS = None
    CRSError = ValueError
    ProjError = RuntimeError

_EPSG_PATTERN = re.compile(r"\bEPSG\s*[:=]?\s*(\d{3,6})\b", re.IGNORECASE)


def embedded_epsg(value: Any) -> str:
    """Return ``EPSG:<code>`` embedded in a descriptive label, if present."""
    match = _EPSG_PATTERN.search(str(value or ""))
    return f"EPSG:{int(match.group(1))}" if match else ""


def canonical_crs_text(value: Any, *, strict: bool = False) -> str:
    """Return a pyproj-compatible CRS string without discarding display labels.

    Project manifests may contain labels such as
    ``CGCS2000 / 3-degree Gauss-Kruger zone 39 (EPSG:4547)``.  GDAL/pyproj do
    not reliably accept that entire label, so the embedded authority code is
    extracted before falling back to the original text.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    candidates = [text]
    epsg = embedded_epsg(text)
    if epsg and epsg != text.upper():
        candidates.append(epsg)
    if CRS is not None:
        for candidate in candidates:
            try:
                return CRS.from_user_input(candidate).to_string()
            except (CRSError, ProjError, TypeError, ValueError):
                continue
    elif epsg:
        return epsg
    if strict:
        raise ValueError(f"无法解析坐标系：{text}")
    return epsg or text


def describe_crs(value: Any) -> tuple[str, str]:
    """Return an authority label and central meridian for display.

    Unknown values remain explicit instead of substituting a project-specific
    demonstration CRS.
    """
    text = str(value or "").strip()
    if not text:
        return "未配置", "--"
    normalized = canonical_crs_text(text)
    if CRS is None:
        return embedded_epsg(text) or normalized or text, "--"
    try:
        crs = CRS.from_user_input(normalized or text)
    except (CRSError, ProjError, TypeError, ValueError):
        return embedded_epsg(text) or text, "--"
    authority = crs.to_authority()
    authority_text = f"{authority[0]}:{authority[1]}" if authority else (embedded_epsg(text) or crs.to_string())
    central = "--"
    operation = crs.coordinate_operation
    if operation is not None:
        for parameter in operation.params:
            name = str(parameter.name or "").lower()
            if "longitude of natural origin" in name or "central meridian" in name:
                value_deg = float(parameter.value)
                direction = "E" if value_deg >= 0 else "W"
                absolute = abs(value_deg)
                degrees = int(absolute)
                minutes_float = (absolute - degrees) * 60.0
                minutes = int(minutes_float)
                seconds = (minutes_float - minutes) * 60.0
                if abs(seconds - round(seconds)) < 1e-6:
                    seconds_text = f"{int(round(seconds)):02d}"
                else:
                    seconds_text = f"{seconds:05.2f}"
                central = f"{degrees}°{minutes:02d}′{seconds_text}″{direction}"
                break
    return authority_text, central


__all__ = ["canonical_crs_text", "embedded_epsg", "describe_crs"]
