# Dataflow Execution Environment

This is a monorepo with TypeScript. The project uses bun workspaces for package management.

## Build & Run

```bash
# Install
bun install

# Build all packages
bun run --filter '*' build

# Build specific package
bun run --filter './packages/{package-name}' build 

# Development (watch mode)
bun run --filter '*' dev

# Start
bun run --filter '*' start
```

## Validation

Run after implementing to get immediate feedback:

```bash
# Tests (all)
bun test

# Tests (specific package)
bun test ./packages/{package-name}

# Tests (specific layer - Ralph Wiggum method)
bun test -- layer1
bun test -- layer2

# Typecheck
bun run typecheck

# Lint
bun run lint
```

## Operational Notes

### Project Structure

- `packages/` - Contains all workspace packages
- `specs/` - Contains the application specifications

#### Monorepo Structure
- `packages/compiler/` - Parser, validator, type checker
- `packages/runtime/` - Demand-driven evaluator
- `packages/shared/` - Shared types/utilities (project's stdlib)

### Code Standards

- Use TypeScript with strict mode enabled
- Shared code goes in `packages/shared/` with proper exports configuration
- Specs should be split into markdown (*.md) files in `specs/`

### Monorepo Conventions

- Import shared modules using workspace names: `@my-app/shared/example`

### Critical Commands
- Run `bun test` after changes - catches regressions
- Use `bun test -- layer1` to verify current layer before moving on
- Build errors often mean missing inter-package dependencies

### Common Pitfalls
- Don't assume missing - search codebase first
- Temporal operators via cache, NOT mutable state
- Parallelism from Promise.all, NOT pre-calculated levels

### External File Loading

CRITICAL: When you encounter a file reference (e.g., @rules/general.md), use your Read tool to load it on a need-to-know basis. They're relevant to the SPECIFIC task at hand.

Instructions:

- Do NOT preemptively load all references - use lazy loading based on actual need
- Follow references recursively when needed

## Codebase Patterns

### Demand-Driven Evaluation
```typescript
// CORRECT
evaluate(nodeId, time) {
  if (cache.has(nodeId, time)) return cache.get(nodeId, time);
  const inputs = getInputs(nodeId).map(i => evaluate(i.id, time));
  const value = node.evaluate(inputs);
  cache.set(nodeId, time, value);
  return value;
}
```

### Temporal Operators
```typescript
// CORRECT: Pure function of time
case "FBY":
  return (time === 0) 
    ? evaluate(inputs[0].id, time)
    : evaluate(inputs[1].id, time - 1);

// WRONG: Mutable state
// let state = initial; state = next; ❌
```

### Parallelism
```typescript
// CORRECT: Emergent from demands
const vals = await Promise.all(inputs.map(i => evaluate(i.id, time)));

// WRONG: Pre-planned levels
// for (const level of levels) { ... } ❌
```

### Engineering Principles for the Runtime
- Consult 'Lucid, the Dataflow Programming Language.pdf' (in this folder) as one of the theoretical foundations for implementing demand-driven evaluation/lazy evaluation and temporal operators. You may investigate other ways of implementing lazy evaluation but must leave thorough details of it, including links to related documentation for it.

### Git & Commitment Rules
- Follow the [Conventional Commits 1.0.0](https://www.conventionalcommits.org) specification.
- Perform atomic commits. Each logical change must be committed separately before moving to the next task.
- Use lowercase for the description. Do not end the subject line with a period.
- Add 'Co-authored-by: Ralph (OpenCode Agent)' to the commit message body in the footer section.

## Ralph Wiggum Checklist

Before moving to next layer:
- [ ] New functionality fully implemented (no stubs)
- [ ] New tests pass
- [ ] Previous layer tests still pass
- [ ] Typecheck passes
- [ ] Git committed
