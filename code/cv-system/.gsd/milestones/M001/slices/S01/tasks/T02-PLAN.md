# T02: Config loader implementation

**Slice:** S01
**Milestone:** M001

## Goal
Implement load function that reads JSON file, validates with Pydantic, and returns SessionConfig with clear error handling.

## Must-Haves

### Truths
- Loading a valid JSON config returns SessionConfig instance
- Loading invalid JSON (syntax error) raises clear error with file path
- Loading valid JSON but invalid values (e.g., negative FPS) raises Pydantic ValidationError
- Missing file raises FileNotFoundError with helpful message

### Artifacts
- `src/cv_system/config.py` — Contains load_config(path: Path) function (adds to existing file)
- Function uses Path from pathlib
- Returns SessionConfig
- Has docstring explaining behavior and error handling

### Key Links
- T01 → provides SessionConfig model for return type
- T03 → calls load_config() to get configuration

## Steps
1. Add `import Path from pathlib` to config.py
2. Implement `def load_config(path: Path) -> SessionConfig:` function
3. Read JSON file: `path.read_text(encoding="utf-8")`
4. Parse JSON: `json.loads(content)`
5. Validate: `SessionConfig(**data)`
6. Wrap in try/except for FileNotFoundError, json.JSONDecodeError, pydantic.ValidationError
7. Raise custom ConfigError with helpful messages for each error type
8. Add type hints and comprehensive docstring
9. Run `uv run ruff check src/` to verify linting

## Context
- Follow Python exception handling best practices: catch specific exceptions, not bare except
- Use pathlib.Path for cross-platform path handling
- ConfigError can be a simple custom Exception: `class ConfigError(RuntimeError): pass`
- Docstring should mention that path can be relative or absolute
