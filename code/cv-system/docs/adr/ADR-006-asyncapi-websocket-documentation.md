# ADR-006: AsyncAPI for documenting the WebSocket protocol between CV and Language Runtime

## Status

Accepted

## Date

2026-03-19

## Context

The CV system (Python) and the Language Runtime (Bun/Elysia) communicate via WebSocket. This is a bidirectional, asynchronous protocol: the CV system sends detection events (touches, pieces) and the Language Runtime can send commands back (project feedback, request recalibration, etc.). In a previous discussion, LSP was ruled out as an integration protocol for the tangible language because it is designed for text-based languages, and a custom WebSocket protocol was chosen instead.

This protocol needs formal documentation that defines the messages, their schemas, and the communication semantics.

### Alternatives considered

**Informal documentation (Markdown):** Easy to write but not automatically validatable. Message schemas drift from the actual code over time.

**OpenAPI:** Designed for REST APIs (request/response). Does not naturally model bidirectional or event-driven communication.

**AsyncAPI:** A specification designed explicitly for event-driven and asynchronous APIs. Supports WebSocket as a first-class protocol. Allows defining channels, messages, schemas (JSON Schema), and operations (send/receive). Has tooling for documentation generation, schema validation, and code generation.

## Decision

We use **AsyncAPI 3.x** to document the WebSocket protocol between the CV system and the Language Runtime.

### Scope

The document defines servers, channels, messages (with JSON schemas), and operations (send/receive per channel). It lives at the monorepo root (or in `../../../contracts/api`) since it defines the contract between both subsystems.

### Usage

The AsyncAPI document is the source of truth for WebSocket integration. Both subsystems must conform their messages to the schemas defined there. It is a living document updated when message types change.

## Consequences

Learning AsyncAPI syntax is required (low complexity, similar to OpenAPI). The document must be kept up to date when messages change. AsyncAPI Studio (https://studio.asyncapi.com/) can be used to edit and preview. Optionally, the JSON schemas can be used to validate messages at runtime on both sides.
