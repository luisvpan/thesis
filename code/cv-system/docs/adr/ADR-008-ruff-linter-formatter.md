# ADR-008: Ruff as linter and formatter for the CV system

## Status

Accepted

## Date

2026-03-21

## Context

The CV system needs a linter and formatter to enforce consistent code style, catch common errors (unused imports, undefined variables, unreachable code), and maintain code quality as the monolithic file is refactored into multiple modules by multiple contributors.

The inherited code has no linting or formatting configuration. It mixes naming conventions, has unused imports, dead code, and inconsistent style throughout.

### Alternatives considered

**Flake8 + Black + isort:** The traditional Python stack. Requires three separate tools, three configurations, and three invocations. Flake8 is plugin-based (dozens of plugins for different rule sets), which adds dependency management overhead. Black and isort occasionally conflict on import formatting.

**Pylint:** Comprehensive but slow, verbose configuration, and opinionated in ways that often require extensive rule suppression for practical use.

**Ruff:** A single tool (by Astral, same team behind uv) that replaces Flake8, Black, isort, pydocstyle, pyupgrade, and autoflake. Written in Rust, executes 10-100x faster than the alternatives. Supports 900+ lint rules. Configured via `pyproject.toml` alongside the rest of the project. Integrates naturally with uv (`uvx ruff check`, `uvx ruff format`).

## Decision

We use **Ruff** as both linter and formatter for the CV system.

### Configuration

Ruff is configured in `pyproject.toml` under `[tool.ruff]`:

```toml
[tool.ruff]
line-length = 88
indent-width = 4
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # Pyflakes
    "I",    # isort (import sorting)
    "B",    # flake8-bugbear
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
    "RUF",  # Ruff-specific rules
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

The rule set starts conservative (`E`, `F`, `I`, `B`, `UP`, `SIM`, `RUF`) and can be expanded as the team becomes comfortable. Import sorting (`I`) is included from the start to keep imports clean as the codebase is split into modules.

### Installation

Ruff is declared as a dev dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "ruff",
    "pytest",
]
```

It can also be run without installation via `uvx ruff check` and `uvx ruff format`.

## Consequences

A single tool replaces what would otherwise be 3+ separate tools. Configuration lives in the same `pyproject.toml` that uv already uses. Ruff's speed makes it practical to run on every save (via editor integration) and as a pre-commit hook. The formatter (replacing Black) ensures consistent style without debate. Import sorting (replacing isort) keeps imports organized as the module count grows.
