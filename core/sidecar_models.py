#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized sidecar schema constants for RTK/IMU/altimeter parsing."""

from __future__ import annotations

RTK_REQUIRED_FIELDS = ("timestamp_s", "longitude", "latitude")
RTK_OPTIONAL_FIELDS = (
    "ground_elevation_m",
    "flight_height_m",
    "local_x_m",
    "local_y_m",
    "local_z_m",
    "rtk_fix_type",
    "satellites",
    "hdop",
)

IMU_REQUIRED_FIELDS = ("timestamp_s", "roll_deg", "pitch_deg", "yaw_deg")
IMU_OPTIONAL_FIELDS = ("angular_rate_x", "angular_rate_y", "angular_rate_z")

ALTIMETER_REQUIRED_FIELDS = ("timestamp_s", "height_agl_m")
ALTIMETER_OPTIONAL_FIELDS = ("height_source", "snr", "target_count", "valid")

RTK_COLUMN_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "gps_time", "time_s"),
    "longitude": ("longitude", "longitude_deg", "lon", "lng"),
    "latitude": ("latitude", "latitude_deg", "lat"),
    "ground_elevation_m": ("ground_elevation_m", "elevation_m", "altitude_m"),
    "flight_height_m": ("flight_height_m", "height_m", "agl_m"),
    "local_x_m": ("local_x_m", "x_m", "east_m"),
    "local_y_m": ("local_y_m", "y_m", "north_m"),
    "local_z_m": ("local_z_m", "z_m", "up_m"),
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

ALTIMETER_COLUMN_ALIASES = {
    "timestamp_s": ("timestamp_s", "timestamp", "time_s", "height_timestamp_s"),
    "height_agl_m": (
        "height_agl_m",
        "flight_height_m",
        "distance_m",
        "height_m",
        "agl_m",
    ),
    "height_source": ("height_source", "source"),
    "snr": ("snr", "signal_to_noise"),
    "target_count": ("target_count", "targets", "num_targets"),
    "valid": ("valid", "is_valid"),
}
