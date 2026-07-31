# Release process

1. Run the risk-selected merge gate.
2. Run four isolated release shards on Linux and Windows.
3. Build from a clean `.build-venv` using pinned requirements.
4. Generate supply-chain/SBOM evidence and dependency-license inventory.
5. Build the PyInstaller executable with UPX disabled.
6. Run the frozen executable with `--smoke-test`.
7. Sign the executable, publish SHA-256, and preserve gate/build evidence.
8. Verify install, upgrade from supported project schemas, rollback, and uninstall on Windows.
