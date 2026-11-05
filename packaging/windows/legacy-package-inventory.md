# Published 0.11.0 package inventory

`mygpr-0.11.0-managed-files.txt` contains 2,017 SHA-256/path rows captured immediately after installing the official Windows x64 0.11.0 package in a clean disposable GitHub runner. No QA/user files had been added to `_internal`.

- Installer: https://github.com/CYberkra/MyGPR/releases/download/v0.11.0/MyGPR-0.11.0-Windows-x64-Setup.exe
- Installer SHA-256: `2dd386865850996cd242c01021e4105af21c8483cc5af9e6516d9e731130a72b`
- Capture run: https://github.com/CYberkra/MyGPR/actions/runs/42079373526
- Capture source: `5765602`
- Inventory SHA-256: `ee916acc7ac7e6f474db6af186f83150b34221f0bdf7b4de6eba028256b484fc`

The capture run intentionally retains the failing upgrade evidence: that old published installer has no managed-file inventory, and the uncorrected upgrade leaves obsolete package metadata. This document does not claim that run passed.

The installer uses this trusted inventory only when the installed version has no inventory of its own. Cleanup still requires a safe path, an exact current content hash, and absence from the incoming inventory. Unknown or changed files are preserved. Obsolete MyGPR version metadata that contains changed/unknown files blocks installation before mutation; choose a fresh installation directory in that case. Empty ancestor directories are removed with Inno Setup `RemoveDir`, without recursive deletion.
