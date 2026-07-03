#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fake gprMax executable: sleeps, can be timed out/cancelled."""

from __future__ import annotations

import time


def main() -> int:
    print("fake gprmax sleep start")
    time.sleep(8.0)
    print("fake gprmax sleep end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
