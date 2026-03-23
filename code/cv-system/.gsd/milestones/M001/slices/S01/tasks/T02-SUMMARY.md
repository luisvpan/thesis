---
id: T02
parent: S01
milestone: M001
provides:
  - load_config(path: Path) function for loading and validating JSON config files
  - Custom ConfigError exception with clear, actionable error messages
  - Error handling for FileNotFoundError, JSONDecodeError, and ValidationError
requires:
  - slice: S00
    provides: Initial project structure
  - task: T01
    provides: SessionConfig Pydantic model for validation
affects: [S02, S03, S04, S05]
key_files:
  - src/cv_system/config.py (added load_config function and ConfigError)
key_decisions:
  - Use pathlib.Path for cross-platform path handling
  - Provide up to 3 validation errors in summary to avoid overwhelming output
  - Include line numbers and error types in JSON decode errors
patterns_established:
  - ConfigError wraps low-level exceptions with context (file path)
  - All configuration loading goes through load_config() for consistency
drill_down_paths:
  - .gsd/milestones/M001/slices/S01/tasks/T02-PLAN.md
duration: 5min
verification_result: pass
completed_at: 2026-03-22T23:35:00Z
---

# T02: Config loader implementation

**Configuration loader with JSON parsing, Pydantic validation, and comprehensive error handling.**

## What Happened

Implemented `load_config(path: Path)` function that reads JSON configuration files, validates them against SessionConfig, and provides clear error messages for all failure modes:

1. **FileNotFoundError**: Reports the missing file path
2. **JSONDecodeError**: Shows syntax error with line number and message
3. **ValidationError**: Summarizes up to 3 validation errors with field paths and messages

Added custom `ConfigError(RuntimeError)` that wraps low-level exceptions with file path context. Used `pathlib.Path` for cross-platform compatibility. Docstring documents that path can be relative or absolute.

## Deviations
None — followed T02-PLAN.md exactly.

## Files Created/Modified
- `src/cv_system/config.py` — Added ConfigError class and load_config function (30 lines, 194 total)
