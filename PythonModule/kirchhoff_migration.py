#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility façade for native Kirchhoff migration.

Historical imports remain valid while the implementation lives under the
backend infrastructure package.  No numerical logic should be added here.
"""
from mygpr.infrastructure.processing.algorithms.kirchhoff import (
    load_cagpr_kir_parameter_file,
    method_kirchhoff_migration,
)

__all__ = ["load_cagpr_kir_parameter_file", "method_kirchhoff_migration"]
