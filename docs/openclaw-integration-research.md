# OpenClaw Integration Research

Bandito as the model selection engine for OpenClaw's multi-model routing.

## OpenClaw Background

OpenClaw is an open-source, self-hosted personal AI assistant that connects to multiple LLM providers. It has a hierarchical model selection system (`src/agents/model-selection.ts`) with per-agent overrides, fallback chains, and provider credential rotation. Currently, model selection is **static** — configured via JSON with hardcoded task-to-model mappings.

## Integration Options

### Option 1: Fork the `model-router` Skill (Best Path Today)

OpenClaw has an existing [`model-router` skill](https://github.com/openclaw/skills/blob/main/skills/digitaladaption/model-router/SKILL.md) that classifies tasks and routes to models. It uses a three-step process: setup wizard, task classification, model routing via `sessions_spawn`.

**How Bandito replaces static routing:**
- Instead of hardcoded config mapping ("simple -> haiku, complex -> sonnet"), the skill calls `POST /api/v1/bandits/{id}/pull` to get Thompson Sampling's choice
- After task completion, calls `POST /api/v1/bandits/{id}/reward` with a quality/cost signal
- Arms = model options (haiku, sonnet, opus, gpt-4o, etc.)
- The skill uses shell `curl` to hit the Bandito API, then `sessions_spawn` with the chosen model

The existing skill supports three cost-optimization modes (aggressive, balanced, quality). Bandito could replace all three with a single learned policy that adapts over time.

**Pros:** Works today, proven skill architecture, no OpenClaw core changes needed.
**Cons:** Skill-level integration is coarser than a native hook.

### Option 2: `preRequest` Lifecycle Hook (Cleanest, Not Merged Yet)

[PR #12082](https://github.com/openclaw/openclaw/issues/10969) proposes lifecycle hooks for OpenClaw. A [proposal gist](https://gist.github.com/openmetaloom/657c4668c09d235f8da1306e2438904b) (Feb 5, 2026) defines 10 lifecycle phases (5 pre-action, 5 post-action).

The `preRequest` hook can override model selection:

```ts
async preRequest(ctx) {
  const arm = await fetch('https://your-bandito/api/v1/bandits/{id}/pull');
  ctx.request.model = arm.model_name; // e.g. "anthropic/claude-sonnet-4-5"
}
```

A working demo exists (`continuity-plugin` repo), but hooks aren't in OpenClaw core yet.

**Pros:** Cleanest integration, transparent to the rest of the system, per-request granularity.
**Cons:** Not merged yet. Migrate to this when it lands.

### Option 3: Custom Skill from Scratch

Create a `SKILL.md` that wraps Bandito as a [custom skill](https://zenvanriel.nl/ai-engineer-blog/openclaw-custom-skill-creation-guide/). Skills live in a folder with a `SKILL.md` file (YAML frontmatter + natural language instructions). Can be user-invocable (`/optimize-model`) or auto-invoked by the agent.

Skills can integrate with external APIs via shell commands and support injected secrets (`skills.entries.*.env`, `skills.entries.*.apiKey`).

**Pros:** Full control, no dependency on existing skill.
**Cons:** More work than forking `model-router`.

### Option 4: Local HTTP Proxy

Similar to [ClawRouter](https://github.com/BlockRunAI/ClawRouter) — run Bandito as a proxy that intercepts OpenClaw's LLM API calls and routes based on Thompson Sampling. ClawRouter uses 14-dimension weighted scoring; Bandito would use learned arm values instead.

**Pros:** Fully transparent to OpenClaw, works with any config.
**Cons:** Heaviest to set up, another service to run.

## Concept Mapping

| Bandito Concept  | OpenClaw Mapping                                      |
| ---------------- | ----------------------------------------------------- |
| Bandit           | The "model selection" experiment                      |
| Arm              | Each model option (sonnet, haiku, opus, gpt-4o, etc.) |
| Feature vector   | Task signals (complexity, type, token estimate)       |
| Reward           | Quality score, cost efficiency, latency, user rating  |

## What Bandito Needs

A lightweight SDK endpoint that abstracts the pull + arm-to-model mapping:

- `POST /select-model` — accepts task context, returns `{ provider: "anthropic", model: "claude-sonnet-4-5" }` directly
- Wraps `pull_arm` internally and maps arm ID to provider/model string
- Keeps OpenClaw skill simple (one API call, no Bandito internals)

The reward endpoint already works for the feedback loop — the skill calls it after task completion with a composite score.

## Recommendation

1. **Now:** Fork the `model-router` skill. Swap static task-to-model config for Bandito API calls.
2. **When hooks land:** Migrate to a `preRequest` hook plugin for cleaner per-request integration.
3. **Bandito side:** Add a `/select-model` convenience endpoint that returns provider/model strings directly.
