---
estimated_steps: 3
estimated_files: 1
skills_used:
  - lint
  - test
---

# T02: Validate calibrator integration with fixed config

**Slice:** S03 — Calibrator
**Milestone:** M001

## Description

Verify that the attribute name fixes from T01 allow the Calibrator to work correctly with the configuration system. Run the existing test suite for the calibrator to ensure all tests pass, and perform final lint/format validation on the entire calibration module.

## Steps

1. Run `uv run pytest tests/test_calibrator.py -v` to execute all calibrator tests
2. Inspect test output to verify all tests pass (exit code 0)
3. If any tests fail, read `tests/test_calibrator.py` to understand fixture expectations
4. Run `uv run ruff check src/cv_system/config.py src/cv_system/calibration/` to check for lint errors
5. Run `uv run ruff format src/cv_system/config.py src/cv_system/calibration/ --check` to verify formatting
6. If lint or format errors exist, run with --fix to auto-correct
7. Re-run tests after any fixes to confirm everything passes

## Must-Haves

- [ ] All tests in tests/test_calibrator.py pass (no failures)
- [ ] test_calibrator_initialization_valid passes (Calibrator accepts config with new field names)
- [ ] test_calibrator_run_success passes (calibration completes with mock hardware)
- [ ] test_calibrator_run_metadata passes (metadata contains correct field names)
- [ ] ruff check returns exit code 0 (no lint errors)
- [ ] ruff format --check returns exit code 0 (no formatting needed)

## Verification

- Run `uv run pytest tests/test_calibrator.py -v` — all tests pass (exit code 0)
- Run `uv run ruff check src/cv_system/config.py src/cv_system/calibration/` — no errors
- Run `uv run ruff format src/cv_system/config.py src/cv_system/calibration/ --check` — no changes needed

## Observability Impact

- Signals added/changed: None (validation only)
- How a future agent inspects this: Run pytest to verify integration, run ruff to check code quality
- Failure state exposed: Pytest output shows which tests fail and why, ruff shows specific lint errors with line numbers

## Inputs

- `src/cv_system/config.py` — Updated CalibrationConfig with correct field names (from T01)
- `config/session.json` — Updated JSON with correct field names (from T01)
- `tests/test_calibrator.py` — Existing test suite that validates calibrator behavior

## Expected Output

- No code changes (verification only)
- Test execution output showing all tests pass
- Lint/format output showing no issues
