0a. Study `specs/*.md` with up to 500 parallel subagents to learn the application specifications.
0b. Study @IMPLEMENTATION_PLAN.md.
0c. For reference, the application source code is in `packages/*`.

---

1. Your task is to implement functionality per the specifications using parallel subagents. Follow @IMPLEMENTATION_PLAN.md and choose the most important item to address. Before making changes, search the codebase (don't assume not implemented) using subagents. You may use up to 500 parallel subagents for searches/reads and only 1 subagent for build/tests. Use Opus-level subagents when complex reasoning is needed (debugging, architectural decisions).

2. After implementing functionality or resolving problems, run the tests for that unit of code that was improved. If functionality is missing then it's your job to add it as per the application specifications. Ultrathink.

3. When you discover issues, immediately update @IMPLEMENTATION_PLAN.md with your findings using a subagent. When resolved, update and remove the item.

4. When tests are missing from the codebase, implement them according to acceptance criteria and tests specified in @IMPLEMENTATION_PLAN.md and `specs/*.md`.

5. When the tests pass, update @IMPLEMENTATION_PLAN.md, then lint then format the code, then `git add -A` then `git commit` with a message describing the changes. If this fails, let the user know and suggest they do it manually.

---

99999. Important: When authoring documentation, capture the why - tests and implementation importance.

999999. Important: Single sources of truth, no migrations/adapters. If tests unrelated to your work fail, resolve them or document them as part of the increment.

9999999. You may add extra logging if required to debug issues.

99999999. Keep @IMPLEMENTATION_PLAN.md current with learnings using a subagent - future work depends on this to avoid duplicating efforts. Update especially after finishing your turn.

999999999. When you learn something new about how to run the application, update @AGENTS.md using a subagent but keep it brief. For example if you run commands multiple times before learning the correct command then that file should be updated.

9999999999. For any bugs you notice, resolve them or document them in @IMPLEMENTATION_PLAN.md using a subagent even if it is unrelated to the current piece of work.

99999999999. Implement functionality completely. Placeholders and stubs waste efforts and time redoing the same work.

999999999999. When @IMPLEMENTATION_PLAN.md becomes large periodically clean out the items that are completed from the file using a subagent.

9999999999999. If you find inconsistencies in the specs/* then use an Opus-level subagent with ultrathink to update the specs. Specs are living documents - update them when you discover missing details, unclear definitions, or better approaches. Document the spec change in @IMPLEMENTATION_PLAN.md.

99999999999999. IMPORTANT: Keep @AGENTS.md operational only - status updates and progress notes belong in @IMPLEMENTATION_PLAN.md. A bloated AGENTS.md pollutes every future loop's context.

999999999999999. CRITICAL ARCHITECTURE: Follow demand-driven evaluation model from Lucid. Do NOT implement data-driven pipeline or "levels" for parallelism. Parallelism emerges from Promise.all on independent demands.

9999999999999999. CRITICAL ARCHITECTURE: Temporal operators (FBY, NEXT) are NOT mutable state. They are pure functions of time implemented via cache. Study specs/LANGUAGE_SPEC.md if unclear.

99999999999999999. CRITICAL WORKFLOW: Ralph Wiggum method - each layer must be fully functional before moving to next. Never leave a layer half-implemented. Run tests frequently to verify working state.

999999999999999999. IMPORTANT: Every change requires a Conventional Commit as defined in AGENTS.md before finalizing this iteration.