#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized sidecar schema constants for RTK/IMU phase-1 parsing."""

from __future__ import annotations

RTK_REQUIRED_FIELDS = ("timestamp_s", "longitude", "latitude")
RTK_OPTIONAL_FIELDS = (
    "ground_elevation_m",
    "flight_height_m",
    "rtk_fix_type",
    "satellites",
    "hdop",
)

IMU_REQUIRED_FIELDS = ("timestamp_s", "roll_deg", "pitch_deg", "yaw_deg")
IMU_OPTIONAL_FIELDS = ("angular_rate_x", "angular_rate_y", "angular_rate_z")

RTK_COLUMN_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "gps_time", "time_s"),
    "longitude": ("longitude", "lon", "lng"),
    "latitude": ("latitude", "lat"),
    "ground_elevation_m": ("ground_elevation_m", "elevation_m", "altitude_m"),
    "flight_height_m": ("flight_height_m", "height_m", "agl_m"),
    "rtk_fix_type": ("rtk_fix_type", "fix", "fix_type"),
    "satellites": ("satellites", "sat", "num_satellites"),
    "hdop": ("hdop", "dop"),
}

IMU_COLUMN_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "time_s"),
    "roll_deg": ("roll_deg", "roll"),
    "pitch_deg": ("pitch_deg", "pitch"),
    "yaw_deg": ("yaw_deg", "yaw", "heading_deg"),
    "angular_rate_x": ("angular_rate_x", "gyro_x", "gx"),
    "angular_rate_y": ("angular_rate_y", "gyro_y", "gy"),
    "angular_rate_z": ("angular_rate_z", "gyro_z", "gz"),
}
