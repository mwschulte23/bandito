# Next Iteration — Epics & Tasks

Synthesized from: `tasks.md`, `critical_analysis.md`, `planning.md`, `regret-planning.md`, `README.md`, `CLAUDE.md`

---

## Epic 1: Correctness & Quick Wins
> Fix what's broken or misleading before building new things.

- [x] **Fix budget endpoint math** — `budget_used_percent` returns 0.74 when it should return 74%. Remove the 0-100 transform. *(30 min, isolated change)*
- [x] **Handle human reward changes (0→1, 1→0)** — Currently no path to reverse a human reward. Requires recalculating the residual on `state.b` and updating the PATCH flow. *(correctness issue — wrong human feedback permanently corrupts state)*
- [x] **Fix transaction semantics** — `add_event()` uses `flush()` not `commit()`, relying on implicit caller behavior. Audit and make explicit. *(from critical_analysis.md)*
- [x] **Add missing `__init__.py` files** — `app/api/v1/endpoints/`, `app/services/bandit/utils/`. *(trivial)*

---

## Epic 2: Configurable Reward Function
> Hardcoded params block meaningful analysis. Regret numbers are meaningless if sensitivity values are arbitrary.

- [x] **Add reward config to Bandit model** — `cost_sensitivity`, `latency_sensitivity`, `max_cost`, `max_latency` as bandit-level fields with sensible defaults
- [x] **Migrate existing bandits** — Alembic migration backfilling defaults
- [x] **Wire config through `calculate_reward()`** — Replace hardcoded values with bandit-level params
- [x] **Expose in create/update bandit schemas** — Let users set on creation, update later
- [x] **Add `@property` for adjusted reward on event** — `event.adjusted_reward` recomputes from `(immediate_reward, cost, latency)` + bandit config. Avoids storing derived state. *(from tasks.md "rethink" section)*

**Why P1**: Regret analysis (Epic 3) recomputes adjusted rewards. If the sensitivity params are hardcoded and arbitrary, the regret numbers are meaningless. This must land first or concurrently.

---

## Epic 3: Regret Analysis
> The core analytical capability. Proves the system works. See `regret-planning.md` for full design.

- [x] **Resolve open design questions** — See resolved section in `regret-planning.md`
  - ~~Contextualized regret~~: Dropped. Regret = aggregate convergence. Context belongs in arm/temporal analysis.
  - ~~Warm-up exclusion~~: Not needed. Windowing handles it naturally.
  - Arm lifecycle: Oracle = best active arm at compute time. Recalculates on deactivation.
- [x] **Add schemas** — `RegretMetric`, `RegretWindow`, `RegretResponse` in `schemas/bandit_actions.py`
- [x] **Implement `compute_regret()`** — Core logic in `services/bandit/bandit_analysis.py`
  - Accuracy regret (raw reward, best active arm oracle)
  - Adjusted regret (cost/latency-penalized, best active arm oracle)
  - Cost-only and latency-only regret (isolated efficiency metrics)
  - Both immediate and human reward signals
  - Windowed % regret with configurable N
- [x] **Add endpoint** — `GET /{bandit_id}/regret?window_size=30&min_human_events=3` in `api/v1/endpoints/bandit_analysis.py`

**Depends on**: Epic 2 (reward config) — adjusted regret needs real sensitivity params. ✅ Complete

---

## Epic 5: Analysis Revamp
> Modular metric functions that compose into views. Foundation for dashboards and segment analysis.

- [ ] **Decompose analysis into single-metric functions** — Each metric (mean reward, pull count, cost, regret) is a standalone function. Package them into composite responses.
- [ ] **On-the-fly bandit rebuild from event subsets** — Reconstruct state (A, b, theta) from a filtered set of events. Enables "what if we only had these events?" analysis. *(stateless architecture makes this natural)*
- [ ] **Arm pull distribution history** — Historical pull counts over time per arm (replay viz data)
- [ ] **Enrich state analysis endpoint** — Current `get_state_analysis` decomposes theta_hat. Add: per-arm uncertainty trends, convergence indicators, recommended actions.
- [ ] **Shrinkage option** — Regularization on state rebuild for small-sample arms

---

## Epic 6: Segments
> The "killer feature" — per tasks.md. Segments enable context-aware analysis and bandit branching.

- [ ] **Wire segments into pull_arm/reward flow** — Currently segments only work via CRUD event endpoint, not the action endpoints
- [ ] **Segment-level analysis** — "Arm A is best overall but worst on mobile." Aggregate metrics split by segment.
- [ ] **Design bandit branching** — Big architectural decision. Options: bandit family model, cloned bandits, or virtual sub-bandits. Needs design doc before code.
  - Impacts: data model, state management, SDK interface, analysis
  - Consider: parent/child bandit relationship? Shared arms? Independent state?
- [ ] **Implement branching** — After design is settled

**Why P3 not P1**: Segments are powerful but the core loop (pull → reward → analyze) must be solid first. Branching in particular is a large architectural change.

---

## Epic 7: Developer Experience — Python SDK
> How users actually interact with Bandito.

- [ ] **Design discussion: interface patterns** — Decorator? Context manager? Explicit client? Look at patterns from: LangSmith, Weights & Biases, OpenAI SDK
- [ ] **Auto-run experiment mode** — Generate N pulls programmatically, auto-score with LLM, collect human review separately. *(from planning.md — "experiments should be quick")*
- [ ] **Non-bandit comparison mode** — Loop same query through all arms, score identically. Useful for baselines. *(from planning.md)*
- [ ] **Event list filters** — Return events without human reward, without immediate reward, without any reward. Supports human review workflows.
- [ ] **Streamlit dashboard** — Analytics view. *(from tasks.md — "built-in streamlit dashboard?")*

---

## Epic 8: Architecture Cleanup
> Can happen incrementally alongside other epics.

- [ ] **Unified `BanditService` class** — Consolidate `bandit_brain.py`, `bandit_analysis.py`, `data_helper.py` into cohesive service with injected session. Improves testability and transaction boundaries.
- [ ] **Clean commented-out code** — `feature_prep.py:110-143` (input complexity), `rewards.py:39-43`, `auth.py` (Stripe/email)
- [ ] **Route prefix collision** — Three routers share `/bandit` prefix. Assess if this causes issues or is fine with FastAPI's router nesting.

---

## Epic 9: Future Horizon
> Not for this iteration, but captured for context.

- [ ] **Online mode** — Live, continuous optimization vs. experiment mode. *(from planning.md — "not too far off but haven't thought about it")*
- [ ] **JS/TS SDK** — Frontend developer access
- [ ] **Labeled dataset validation** — Run Bandito against text-2-SQL, categorization, or golden datasets to measure real regret. *(from README.md — "true validation comes from a labeled dataset")*
- [ ] **Input complexity features** — Resurrect commented-out Zipf word frequency scoring in `FeatureTransformer`. Would let the bandit learn that different arms handle complex vs. simple queries differently.

---

## Epic 10: Test Suite
> Safety net for confident iteration. Unlocks heavier use of AI coding tools.

- [ ] **Bandit CRUD tests** — Create bandit, add arms, verify state initialization and dimensions
- [ ] **Pull/reward cycle tests** — Pull arm → immediate reward → human reward. Verify state matrices update correctly, event fields populated
- [ ] **Reward math tests** — Unit tests for `calculate_reward()` with various cost/latency combinations
- [ ] **Regret computation tests** — Known scenario with predictable regret curve (e.g., one dominant arm, verify regret → 0)
- [ ] **Feature prep tests** — `FeatureTransformer` produces correct dimensions, one-hot encoding, time features
- [ ] **SDK contract tests** — *(defer until SDK design settles)*

---

## Dependency Graph

```
Epic 1 (Correctness) ✅
  └──→ Epic 2 (Reward Config) ✅
         └──→ Epic 3 (Regret Analysis) ✅
                └──→ Epic 5 (Analysis Revamp)
                       └──→ Epic 6 (Segments)

Epic 7 (SDK/DX) can start after Epic 3
Epic 8 (Cleanup) is incremental, anytime
Epic 9 (Future) depends on everything
Epic 10 (Tests) lowest priority, anytime
```
