# RECOMMENDED TOOLS - Skills & MCPs

## Model Context Protocol (MCP) Servers

### Essential for This Project

#### 1. **Filesystem MCP**
**Purpose:** Read/write project files, search codebase
**Why needed:** Core to Ralph Wiggum method - agent needs to read specs, modify code, update plans
**Configuration:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dataflow-language"]
    }
  }
}
```

#### 2. **Git MCP**
**Purpose:** Commit changes, create tags, manage branches
**Why needed:** Automatic commits after passing tests, tagging stable versions
**Configuration:**
```json
{
  "mcpServers": {
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "/path/to/dataflow-language"]
    }
  }
}
```

#### 3. **GitHub MCP**
**Purpose:** Create issues, PRs, manage project board
**Why needed:** Document discovered bugs, track implementation progress
**Configuration:**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your_token>"
      }
    }
  }
}
```

---

### Recommended for Enhanced Workflow

#### 4. **Sequential Thinking MCP**
**Purpose:** Break down complex problems into steps
**Why needed:** Planning phase benefits from structured thinking for complex features
**Use case:** Designing temporal operator implementation, architecture decisions
**Configuration:**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

#### 5. **Memory MCP**
**Purpose:** Persistent knowledge graphs across sessions
**Why needed:** Remember architectural decisions, tried approaches, what didn't work
**Use case:** Avoid re-trying failed implementations, recall why certain design choices were made
**Configuration:**
```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

#### 6. **PostgreSQL MCP** (Optional)
**Purpose:** Store test results, benchmarks, metrics over time
**Why needed:** Track performance regression across layers
**Use case:** Store benchmark results per git tag, compare performance across versions
**Configuration:**
```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/dataflow_metrics"]
    }
  }
}
```

#### 7. **Brave Search MCP** (Optional)
**Purpose:** Search web for TypeScript patterns, Jest best practices
**Why needed:** Look up current best practices when stuck
**Configuration:**
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "<your_api_key>"
      }
    }
  }
}
```

---

## Claude Skills (Custom Instructions)

### Essential Skills to Create

#### 1. **TypeScript Monorepo Skill**
**Path:** `~/.claude/skills/typescript-monorepo/`
**Contents:**
```markdown
# TypeScript Monorepo Best Practices

When working in TypeScript monorepos:

1. Use workspace references in tsconfig.json
2. Build dependencies before dependents
3. Shared types go in packages/shared/
4. Each package has own tsconfig.json extending root
5. Use npm workspaces, not lerna/nx for this project

Common commands:
- `npm run build --workspace=packages/compiler`
- `npm test --workspace=packages/runtime`
```

#### 2. **Demand-Driven Evaluation Skill**
**Path:** `~/.claude/skills/demand-driven-dataflow/`
**Contents:**
```markdown
# Demand-Driven Dataflow Implementation

Key principles from Lucid:

1. Only evaluate when DEMANDED
   - NOT when data becomes available
   - NOT in pre-planned "levels"

2. Cache all computed values
   - Cache key: (nodeId, time)
   - Check cache before evaluating

3. Recursive demand propagation
   - evaluate(nodeId) calls evaluate(inputId)
   - Forms dynamic call tree

4. Temporal operators are PURE FUNCTIONS
   - FBY(X, Y) at time t: 
     * t=0 → evaluate X at t=0
     * t>0 → evaluate Y at t-1
   - NO mutable state variables

Anti-patterns to avoid:
- ❌ Pre-calculating execution "levels"
- ❌ Data-driven pipeline (push model)
- ❌ Mutable state for temporal operators
```

#### 3. **Ralph Wiggum Layer Implementation Skill**
**Path:** `~/.claude/skills/ralph-wiggum-method/`
**Contents:**
```markdown
# Ralph Wiggum Incremental Implementation

Rules for layer-based development:

1. ONE concept per layer
   - Layer 1: Only Natural + ADD
   - Layer 2: Only add arithmetic ops
   - Don't mix concepts

2. Layer must be COMPLETE before moving on
   - All tests passing
   - Type checking passing
   - Git tag created
   - Previous layers still working

3. No placeholders or stubs
   - If you add a function, implement it fully
   - No `// TODO: implement this`
   - Better to not add it than to stub it

4. Test-driven layer completion
   - Write tests from acceptance criteria FIRST
   - Implement until tests pass
   - Refactor if needed
   - Don't move on with failing tests

Checklist before moving to next layer:
- [ ] All new functionality fully implemented
- [ ] New tests pass
- [ ] Old tests STILL pass (regression check)
- [ ] Type checking passes
- [ ] Git committed and tagged
```

#### 4. **Acceptance Criteria Writing Skill**
**Path:** `~/.claude/skills/acceptance-criteria/`
**Contents:**
```markdown
# Writing Effective Acceptance Criteria

Format (required):

```markdown
## Acceptance Criteria: [Feature Name]

### Behavioral Outcomes
- ✓ What the system DOES (observable)
- ✓ Focus on WHAT, not HOW

### Observable Results  
- Input X → Output Y (concrete examples)
- Edge case → Expected behavior

### Performance Requirements
- Latency < Nms
- Throughput > M ops/sec

### Edge Cases
- List edge cases + expected behavior
```

Good examples:
- ✓ "Returns empty set when filtering with no matches"
- ✓ "Evaluates each node maximum once per timestep"
- ✓ "Completes evaluation in <5ms for 100-node graph"

Bad examples (too prescriptive):
- ❌ "Uses HashMap for caching with LRU policy"
- ❌ "Implements using visitor pattern"
- ❌ "Stores state in private field _cache"

Principle: Specify WHAT to verify, not HOW to implement.
```

---

### Recommended Skills to Explore

#### 5. **Jest Testing Patterns Skill**
**Purpose:** Best practices for testing demand-driven systems
**Key patterns:**
- Mocking time for temporal operators
- Testing cache behavior
- Benchmarking performance
- Snapshot testing for graph structures

#### 6. **TypeScript Type System Skill**
**Purpose:** Advanced typing for dataflow graphs
**Key patterns:**
- Discriminated unions for node types
- Generic types for Set<T> and Stream<T>
- Type guards for runtime type checking
- Branded types for node IDs

---

## Integration with Prompts

### How MCPs Support the Workflow

**Planning Phase (PROMPT_plan.md):**
- Filesystem MCP: Read specs, search codebase
- Sequential Thinking MCP: Break down complex features
- Memory MCP: Recall previous planning decisions
- GitHub MCP: Check existing issues before creating duplicates

**Building Phase (PROMPT_build.md):**
- Filesystem MCP: Modify code, run tests
- Git MCP: Commit after passing tests, create tags
- PostgreSQL MCP: Store benchmark results
- Brave Search MCP: Look up syntax when stuck

**Skills Integration:**
- Skills provide **domain knowledge** (demand-driven evaluation)
- MCPs provide **capabilities** (read files, commit code)
- Together: Skills guide HOW to use MCPs effectively

---

## Setup Instructions

### 1. Install MCP Servers
```bash
# Install globally or add to project
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-git  
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-sequential-thinking
npm install -g @modelcontextprotocol/server-memory
```

### 2. Configure Claude Desktop
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/projects/dataflow-language"]
    },
    "git": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-git", "/Users/you/projects/dataflow-language"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### 3. Create Skills
```bash
mkdir -p ~/.claude/skills
cd ~/.claude/skills

# Create each skill directory with SKILL.md
mkdir typescript-monorepo demand-driven-dataflow ralph-wiggum-method acceptance-criteria
```

### 4. Verify Setup
In Claude Desktop:
- Check "Tools" menu shows MCP servers
- Create a new project
- Verify skills appear in suggestions

---

## Alternative: Non-Claude Agents

If using other LLM agents (Cursor, Cline, Aider, etc.):

### Filesystem Access
Most agents have native filesystem access, no MCP needed.

### Git Integration
- **Cursor:** Has built-in git
- **Cline:** Use bash commands (`git commit`, `git push`)
- **Aider:** Has --auto-commits flag

### Skills/Knowledge
For agents without skill system:
- Embed key patterns in PROMPT_*.md directly
- Reference pattern docs in prompts
- Use "study these principles before implementing" sections

### Sequential Thinking
Not available in non-Claude agents. Alternatives:
- Chain-of-thought prompting
- "Think step by step" instructions
- Explicit breakdown requests in prompts

---

## Cost Optimization

### MCP Server Costs
Most MCP servers are free (run locally):
- ✅ Filesystem, Git, Sequential Thinking: Free
- 💰 GitHub: Free (needs personal access token)
- 💰 Brave Search: $5/month for API access
- 💰 PostgreSQL: Free (self-hosted) or $15-50/month (managed)

### Skill Costs
Skills are free (local instruction files).

### Recommendation for MVP
**Essential (Free):**
- Filesystem MCP
- Git MCP
- TypeScript Monorepo Skill
- Demand-Driven Evaluation Skill
- Ralph Wiggum Method Skill

**Skip for MVP:**
- Brave Search (use web_search tool in prompts instead)
- PostgreSQL (use JSON files for metrics)
- Memory MCP (use IMPLEMENTATION_PLAN.md instead)

---

## Custom MCP Server Ideas

### DataflowLanguage MCP (Future)
Custom MCP server for project-specific operations:

**Tools:**
- `validate_program` - Validate JSON against schema
- `run_program` - Execute dataflow program
- `benchmark_operation` - Time specific operations
- `suggest_tests` - Generate tests from acceptance criteria

**Implementation:**
```typescript
// packages/mcp-server/src/index.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

const server = new Server({
  name: "dataflow-language",
  version: "1.0.0"
});

server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "validate_program") {
    const compiler = new Compiler();
    const result = compiler.compile(request.params.arguments.program);
    return { content: [{ type: "text", text: JSON.stringify(result) }] };
  }
  // ... other tools
});
```

**Usage in prompts:**
```
Use the dataflow-language MCP server to validate this program before implementing:
[calls validate_program tool]
```

---

**Document Status:** Recommendations based on project needs
**Last Updated:** Initial recommendations
**Next Steps:** Install essential MCPs, create core skills
