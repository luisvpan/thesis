# ADR-002: uv as the package and environment manager for the Python system

## Status

Accepted

## Date

2026-03-19

## Context

The CV system needs a Python dependency manager. The project lives in a monorepo alongside the Language Runtime (TypeScript/Bun). We need a tool that handles virtual environments, dependency resolution, and lockfiles reproducibly.

OpenNI2 is a non-standard dependency not available on PyPI. It is installed as a native system library and consumed via Python bindings that require a path to the redistributable directory (`Redist/`).

## Decision

We use **uv** (by Astral) as the Python project manager.

### Project structure

The CV system lives in `cv-system/` within the monorepo, with its own `pyproject.toml` and `uv.lock`. The structure is independent from the TypeScript side: each system manages its dependencies with its own tool (`uv` for Python, `bun` for TypeScript).

### OpenNI2 handling

OpenNI2 is not declared as a dependency in `pyproject.toml`. It is treated as a system prerequisite, documented in the system's README and in a `.env.example` file. The path to the redistributable binaries is configured via the `OPENNI2_REDIST_PATH` environment variable. The hardware module (`hardware/kinect.py`) reads this variable at initialization and fails with a clear message if it is not defined.

If the OpenNI2 Python binding is available via pip (the `primesense` package), it is included as a normal dependency. If it is a local binding bundled with the SDK installation, it is documented as a prerequisite and the system startup verifies its availability with an import guard.

### Entry scripts

The `pyproject.toml` defines an entry script:

```toml
[project.scripts]
cv-system = "cv_system:main"
```

This allows starting the system with `uv run cv-system`.

## Consequences

Each developer needs to install OpenNI2 manually and configure `OPENNI2_REDIST_PATH` before running the system. This is documented in the README and validated at startup. All other dependencies (OpenCV, NumPy, websockets) are installed automatically with `uv sync`.
