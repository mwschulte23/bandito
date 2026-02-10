# Next Iteration — Epics & Tasks

Synthesized from: `tasks.md`, `critical_analysis.md`, `planning.md`, `README.md`, `CLAUDE.md`

---

## Epic 1: Correctness & Quick Wins ✅
> Fix what's broken or misleading before building new things.

- [x] **Fix budget endpoint math**
- [x] **Handle human reward changes (0→1, 1→0)**
- [x] **Fix transaction semantics**
- [x] **Add missing `__init__.py` files**

---

## Epic 2: Configurable Reward Function ✅
> Hardcoded params block meaningful analysis.

- [x] **Add reward config to Bandit model**
- [x] **Migrate existing bandits**
- [x] **Wire config through `calculate_reward()`**
- [x] **Expose in create/update bandit schemas**
- [x] **Add `@property` for adjusted reward on event**

---

## Epic 3: Regret Analysis ✅
> Core analytical capability. Proves the system works.

- [x] **Resolve open design questions**
- [x] **Add schemas** — `RegretMetric`, `RegretWindow`, `RegretResponse`
- [x] **Implement `compute_regret()`** — accuracy, adjusted, cost-only, latency-only; both immediate and human signals
- [x] **Add endpoint** — `GET /{bandit_id}/regret`

Docs: `docs/regret-analysis.md`, `docs/analysis-endpoints.md`

---

## HOTFIX: State Corruption on Arm Addition 🚨
> Adding arms to an existing bandit can silently corrupt the entire learned state.

**Root cause:** `FeatureTransformer` sorts model/prompt names alphabetically to build feature vectors (`feature_prep.py:21-22`), but `BanditState.resize()` appends new dimensions to the END of existing arrays (`models/bandit.py:99-127`). When a new arm introduces a model or prompt that sorts *before* existing ones, the feature index mapping shifts but the state arrays don't remap.

**Example:** Bandit has `model_B` (feature index 0). Add `model_A` → sorted order becomes `[model_A, model_B]`. Now `model_A` occupies index 0 and inherits `model_B`'s learned weights. `model_B` shifts to index 1 and picks up what was the prompt weight. The entire A matrix, b vector, theta_hat, and Cholesky become misaligned with the new feature ordering.

**Affected code:**
- `app/services/bandit/utils/feature_prep.py` — `FeatureTransformer.__init__()` sorts models/prompts
- `app/models/bandit.py` — `BanditState.resize()` assumes new dims go at end
- `app/api/v1/endpoints/bandit.py:127-131` — arm creation triggers resize

**Fix options (pick one):**
1. **Remap on resize** — When dimensions change, compute old→new index mapping and reorder A, b, theta, chol accordingly
2. **Stable feature ordering** — Use arm creation order (or arm ID order) instead of sorted names, so new arms always get appended indices
3. **Full state rebuild** — On arm addition, replay all events to reconstruct state from scratch (safest but expensive)

- [ ] **Fix feature index ↔ state dimension alignment on arm addition**
- [ ] **Add regression test** — Add arm that sorts before existing, verify state integrity

---

## Epic 3.5: Exploration Improvements
> Pure Thompson Sampling has no cold-start safety net. First N pulls should guarantee every arm gets tried.

- [ ] **Add forced exploration for first N pulls per arm** — Before TS kicks in, cycle through arms uniformly (or round-robin) so every arm gets a minimum observation count. Configurable `min_pulls_per_arm` on the Bandit model (default 3). `pull_arm` checks pull counts and forces under-explored arms until the threshold is met. After that, standard TS with β=1.0 takes over. β=1.0 is already sound — it over-explores relative to the true posterior (the update rule assumes σ²=1, but real reward noise is σ²≈0.04–0.25), so ongoing exploration after the forced period is naturally generous.

---

## Epic 4: Python SDK 🔜
> How users actually interact with Bandito. This is the priority.

- [ ] **Design discussion: interface patterns** — Decorator? Context manager? Explicit client? Look at patterns from: LangSmith, Weights & Biases, OpenAI SDK
- [ ] **Auto-run experiment mode** — Generate N pulls programmatically, auto-score with LLM, collect human review separately. *(experiments should be quick to set up)*
- [ ] **Non-bandit comparison mode** — Loop same query through all arms, score identically. Useful for baselines.
- [ ] **Event list filters** — Return events without human reward, without immediate reward, without any reward. Supports human review workflows.
- [ ] **Streamlit dashboard** — Analytics view

---

## Epic 5: Analysis Revamp
> Modular metric functions that compose into views. Foundation for dashboards and segment analysis.

- [ ] **Decompose analysis into single-metric functions** — Each metric is a standalone function. Package into composite responses.
- [ ] **On-the-fly bandit rebuild from event subsets** — Reconstruct state from filtered events. Enables "what if?" analysis.
- [ ] **Arm pull distribution history** — Historical pull counts over time per arm
- [ ] **Enrich state analysis endpoint** — Per-arm uncertainty trends, convergence indicators, recommended actions.
- [ ] **Shrinkage option** — Regularization on state rebuild for small-sample arms

---

## Epic 6: Test Suite
> Safety net for confident iteration.

- [ ] **Bandit CRUD tests** — Create bandit, add arms, verify state initialization
- [ ] **Pull/reward cycle tests** — Pull → immediate reward → human reward. Verify state updates.
- [ ] **Reward math tests** — Unit tests for `calculate_reward()` with various cost/latency combos
- [ ] **Regret computation tests** — Known scenario with predictable regret curve
- [ ] **Feature prep tests** — `FeatureTransformer` produces correct dimensions
- [ ] **SDK contract tests** — *(defer until SDK design settles)*

---

## Parked — Will Spec Later

These are real but not next. Will get deeper design docs when the time comes.

- **Segments** — Per-context analysis, bandit branching, segment-level metrics
- **Architecture Cleanup** — Unified `BanditService`, commented code cleanup, route prefix audit
- **Future Horizon** — Online mode, JS/TS SDK, labeled dataset validation, input complexity features

---

## Dependency Graph

```
Epic 1 (Correctness) ✅
  └──→ Epic 2 (Reward Config) ✅
         └──→ Epic 3 (Regret Analysis) ✅
                └──→ HOTFIX (State Corruption) ← DO THIS FIRST
                └──→ Epic 3.5 (Exploration)
                └──→ Epic 4 (SDK) ← YOU ARE HERE
                       └──→ Epic 5 (Analysis Revamp)

Epic 6 (Tests) anytime, grows with codebase
```
