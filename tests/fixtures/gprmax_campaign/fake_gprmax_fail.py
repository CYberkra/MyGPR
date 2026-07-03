#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fake gprMax executable: fail."""

from __future__ import annotations

import sys


def main() -> int:
    print("fake gprmax fail stderr", file=sys.stderr)
    return 7


if __name__ == "__main__":
    raise SystemExit(main())
