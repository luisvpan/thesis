0a. Study @PROJECT_GOALS.md to learn the project goals.
0b. Study `specs/*.md` with up to 250 parallel subagents to learn the application specifications.
0c. Study @IMPLEMENTATION_PLAN.md (if present) to understand the plan so far.
0d. Study `packages/shared/*` with up to 250 parallel subagents to understand shared utilities & components.
0e. For reference, the application source code is in `packages/*`.

---

## Requirements Exploration (If Specs Are Incomplete)

TRIGGER: When specs are minimal, vague, or missing acceptance criteria.

Use AskUserQuestionTool (or equivalent) to conduct structured exploration:

### Acceptance Criteria
For each feature ask:
- "How will we know this feature works correctly?"
- "What observable behaviors indicate success?"
- "What performance requirements exist?"

### Multiple Valid Approaches
When encountering design choices:
- "For temporal operators: Lucid's tagging approach or explicit state?"
- "For type checking: Strict compile-time or permissive with runtime checks?"
- "For parallelism: Promise.all only or also Workers?"

After exploration, create/update specs with acceptance criteria format:

```markdown
## Acceptance Criteria: [Feature Name]

### Behavioral Outcomes
- ✓ Observable behavior (WHAT system does)

### Observable Results
- Input X → Output Y

### Performance Requirements
- Latency < Nms

### Edge Cases
- Case → Expected behavior
```

---

1. Study @IMPLEMENTATION_PLAN.md (if present; it may be incorrect) and use up to 500 subagents to study existing source code in `packages/*` and compare it against `specs/*`. Use an Opus-level subagent to analyze findings, prioritize tasks, and create/update @IMPLEMENTATION_PLAN.md as a bullet point list sorted in priority of items yet to be implemented. Ultrathink. Consider searching for TODO, minimal implementations, placeholders, skipped/flaky tests, and inconsistent patterns. Study @IMPLEMENTATION_PLAN.md to determine starting point for research and keep it up to date with items considered complete/incomplete using subagents.

2. For each planned feature, derive test requirements from acceptance criteria in specs. Add to @IMPLEMENTATION_PLAN.md:

```markdown
## Feature: [Name]

### From Acceptance Criteria (specs/FILE.md):
- ✓ Behavioral outcome 1
- ✓ Edge case X

### Required Tests:
- [ ] Test: Behavioral outcome 1
- [ ] Test: Edge case X
- [ ] Benchmark: Performance requirement
```

IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume functionality is missing; confirm with code search first. Treat `packages/shared/` as the project's standard library for shared utilities and components. Prefer consolidated, idiomatic implementations there over ad-hoc copies.

ULTIMATE GOAL: We want to achieve a working dataflow programming language for 6-9 year old children that executes tangible block programs with correct demand-driven semantics and real-time AR feedback. Study @PROJECT_GOALS.md to learn the project goals.

Consider missing elements and plan accordingly. If an element is missing, search first to confirm it doesn't exist, then if needed author the specification at specs/FILENAME.md with acceptance criteria. If you create a new spec then document the plan to implement it in @IMPLEMENTATION_PLAN.md using a subagent.

---

99999. When authoring specifications, capture the why - not just what to build, but educational rationale.

999999. Acceptance criteria focus on behavioral outcomes (observable results), NOT implementation approaches (technical decisions).

9999999. Test requirements in @IMPLEMENTATION_PLAN.md must map to acceptance criteria in specs. No tests that don't verify real requirements.

99999999. Single sources of truth. Specs = WHAT to achieve. Plan = HOW to verify + priority order.

999999999. Keep @IMPLEMENTATION_PLAN.md current with learnings using a subagent.

9999999999. When @IMPLEMENTATION_PLAN.md becomes large, periodically clean out completed items using a subagent.

99999999999. For any bugs or inconsistencies discovered, document in @IMPLEMENTATION_PLAN.md using a subagent.

999999999999. If you find inconsistencies in specs/* then use an Opus-level subagent with ultrathink to update the specs. Specs (including LANGUAGE_SPEC.md) are living documents that evolve as you discover edge cases, implementation challenges, or better designs. When updating a spec, ensure acceptance criteria remain clear and behavioral.

9999999999999. IMPORTANT: Keep @AGENTS.md operational only - status updates and progress notes belong in @IMPLEMENTATION_PLAN.md. A bloated AGENTS.md pollutes every future loop's context.
