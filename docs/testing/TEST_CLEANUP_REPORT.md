# Test-system cleanup report

## Scope

The existing regression suite was retained. Cleanup was limited to evidence-backed redundancy and taxonomy problems; behavior tests were not discarded merely because they inspect related code paths.

## Changes

- Removed five exact-duplicate compatibility-presence test functions from five AutoTune runner modules.
- Replaced them with one canonical compatibility-freeze contract in `tests/industrial/static_contract/test_frozen_compatibility_modules_v1.py`.
- Added an exact AST-body duplicate gate; duplicate groups must remain zero.
- Eliminated the `misc` population from 70 modules to zero through explicit domain classifications and a declared `legacy_regression` fallback.
- Separated source/configuration assertions conceptually as `static_contract`; they are no longer counted as behavioral evidence in the industrial acceptance layer.
- Added seven industrial modules covering acceptance, reliability, performance, scientific validation, properties/fuzzing, and test governance.
- Added a deterministic real-data subset for all six Yingshan lines; full raw files remain external immutable assets referenced by SHA-256.

## Inventory after cleanup

- Test modules: 290
- Static test functions: 1,517
- Exact duplicate test-body groups: 0
- Unclassified (`misc`) modules: 0
- Industrial test modules: 7
- Yingshan full-file hashes: 6 verified

## Retained intentionally

Source-text contract tests were not deleted wholesale. They remain useful for architecture boundaries, packaging declarations, dependency bans, and migration freezes. They are classified as static contracts or legacy regression and cannot substitute for the runtime acceptance tests added in this phase.
