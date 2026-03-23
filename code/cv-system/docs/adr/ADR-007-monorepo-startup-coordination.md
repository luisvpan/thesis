# ADR-007: Monorepo startup coordination

## Status

Accepted

## Date

2026-03-19

## Context

The project is a monorepo with two systems in different languages: the Language Runtime (TypeScript/Bun) and the CV System (Python/uv). Both need to be running and coordinated during a real session with users. During development, it is common to need only one of the two running.

### Alternatives considered

**Turborepo / nx:** Powerful but adds another tool to the stack with significant configuration and learning curve for a two-system project.

**Docker Compose:** Adds containerization complexity (Dockerfiles, volumes for the Kinect, USB device access) that is non-trivial for hardware like the Kinect.

**Makefile / justfile:** Simple tools that define targets for starting systems individually or together. No external dependencies, transparent, easy to maintain.

## Decision

We use a **Makefile** (or `justfile` if the team prefers) at the monorepo root to coordinate startup.

### Connection coordination

The CV system connects to the Language Runtime via WebSocket. The URL is configured via `LANGUAGE_RUNTIME_WS_URL` (default: `ws://localhost:3000/cv`). Startup order does not matter: the CV system retries with exponential backoff. The Language Runtime accepts connections without requiring the CV system to be connected.

This allows developing each system independently.

## Consequences

The solution is intentionally simple. If the project grows in complexity, it can be migrated to Docker Compose or Turborepo without changes to individual systems. The retry-based connection coordination makes the system resilient to startup order.
