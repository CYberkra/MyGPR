"""Frozen launcher and an explicit package-verification command for release QA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def package_check(report_path: Path, hdf5_path: Path | None) -> int:
    report = {"frozen": bool(getattr(sys, "frozen", False)), "status": "failed"}
    try:
        import importlib.metadata
        import numpy as np
        import pyproj
        import rasterio
        import fiona
        from core.field_report_export import _software_version
        from mygpr.interfaces.cli.backend_smoke import run_smoke
        from ui.pages.settings_page import _read_version

        versions = {"distribution": importlib.metadata.version("mygpr"),
                    "report": _software_version(), "settings": _read_version()}
        assert len(set(versions.values())) == 1, versions
        report["versions"] = versions
        report["backend"] = run_smoke(include_autotune=True)
        assert report["backend"]["finite"]
        xy = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(104.0, 30.0)
        assert np.isfinite(xy).all()
        with rasterio.Env() as environment:
            assert "GTiff" in environment.drivers()
        with fiona.Env() as environment:
            assert "GeoJSON" in environment.drivers()
        report["geospatial"] = "passed"
        if hdf5_path is not None:
            import h5py
            from core.gpr_io import auto_load_data

            def source_hash():
                with hdf5_path.open("rb") as handle:
                    return hashlib.file_digest(handle, "sha256").hexdigest()

            before = source_hash()
            loaded = auto_load_data(str(hdf5_path))
            matrix = loaded["data"]
            try:
                data = np.asarray(matrix)
                with h5py.File(hdf5_path, "r") as handle:
                    np.testing.assert_array_equal(data, handle["traces"][:].T)
                    header = loaded['header_info']
                    if 'time_axis_endpoint' in handle.attrs:
                        assert header['time_axis_endpoint'] == bool(handle.attrs['time_axis_endpoint'])
                    np.testing.assert_allclose(header['total_time_ns'], handle.attrs['time_window_ns'])
                    divisor = max(1, data.shape[0] - int(header['time_axis_endpoint']))
                    np.testing.assert_allclose(header['time_step_s'], header['total_time_ns'] * 1e-9 / divisor)
                    field_names = {
                        'lon': 'longitude', 'lat': 'latitude', 'alt': 'altitude_m', 'height': 'flight_height_m',
                        'roll': 'roll_deg', 'pitch': 'pitch_deg', 'yaw': 'yaw_deg',
                        'ax': 'accel_x_g', 'ay': 'accel_y_g', 'az': 'accel_z_g',
                        'gx': 'angular_rate_x_dps', 'gy': 'angular_rate_y_dps', 'gz': 'angular_rate_z_dps',
                        'timestamp_ms': 'timestamp_ms', 'trace_id': 'trace_id',
                    }
                    metadata = loaded['trace_metadata']
                    for source_name, target_name in field_names.items():
                        field = 'trace_meta/' + source_name
                        if field in handle:
                            expected = handle[field][:]
                            if source_name == 'trace_id':
                                assert metadata[target_name].dtype == np.dtype('uint64')
                            else:
                                expected = np.where(np.isfinite(expected), expected, np.nan)
                            np.testing.assert_array_equal(metadata[target_name], expected)
                    if 'timestamp_ms' in metadata:
                        np.testing.assert_array_equal(metadata['trace_timestamp_s'], metadata['timestamp_ms'] / 1000.)
                report["hdf5"] = {
                    "shape": list(data.shape),
                    "time_window_ns": loaded["header_info"]["total_time_ns"],
                    "distance_axis_status": loaded["header_info"]["distance_axis_status"],
                    "time_axis_endpoint": loaded["header_info"]["time_axis_endpoint"],
                    "time_step_s": loaded['header_info']['time_step_s'],
                    "trace_metadata_verified": True,
                    "missing_sensor_values": all(np.isnan(metadata[name]).all() for name in field_names.values()
                                                 if name in metadata and name not in ('trace_id', 'timestamp_ms')),
                }
                assert report["hdf5"]["distance_axis_status"] == "unknown"
            finally:
                close = getattr(matrix, "close", None)
                if close is not None:
                    close()
            assert source_hash() == before
            report["hdf5"]["source_unchanged"] = True
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name("." + report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    # Tests may isolate their settings without changing HOME/USERPROFILE or user data.
    settings_file = os.environ.get("MYGPR_SETTINGS_FILE")
    if settings_file:
        from ui import constants
        constants.SETTINGS_FILE = settings_file
        constants.SETTINGS_DIR = str(Path(settings_file).parent)
    if "--package-check" in sys.argv[1:]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--package-check", required=True, type=Path)
        parser.add_argument("--hdf5", type=Path)
        args = parser.parse_args()
        result = package_check(args.package_check, args.hdf5)
        # Qt libraries can crash during interpreter shutdown. Preserve the test result
        # only after its assertions and atomic evidence write have completed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(result)
    from app_qt import main as gui_main
    return gui_main()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
