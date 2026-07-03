#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fake gprMax executable: success."""

from __future__ import annotations

import sys


def main() -> int:
    print("fake gprmax success stdout")
    print(f"args={sys.argv[1:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
