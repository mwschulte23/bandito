# Regret Analysis — Implementation Plan ✅ COMPLETE

## Overview

Add regret computation to `bandit_analysis.py` exposing two regret types, each computed for both reward signals (immediate + human), returned as a windowed time series of % regret.

> **Status**: Implemented in Epic 3. Four metrics (accuracy, adjusted, cost-only, latency-only) with both immediate and human reward signals. Generic `compute_oracle()` + `compute_window_regret()` core functions accept any reward extraction function.

---

## Two Regret Types

### 1. Accuracy Regret
- **Question answered**: "Are we converging on the highest quality arm?"
- **Reward signal**: `immediate_reward` (raw quality) or `human_reward` (raw binary)
- **Oracle**: Best **active** arm — the active arm with the highest mean raw reward
- **No cost/latency adjustment**

### 2. Adjusted Regret
- **Question answered**: "Are we converging on the best *value* arm?"
- **Reward signal**: `calculate_reward(immediate_reward, cost, latency)` — recomputed from stored fields
- **Oracle**: Best **active** arm by mean adjusted reward
- **Captures cost/latency tradeoffs** — a slightly worse arm that's cheaper/faster may be optimal here

### Diagnostic: Divergence Between Layers
- Accuracy low, Adjusted high → best quality arm is expensive/slow, system correctly trading off
- Both low → converged, Thompson Sampling is working

### Why Not Contextualized Regret?
Regret is an aggregate convergence measure — "is the system finding the best arm overall?" Per-context performance analysis (temporal patterns, feature-based arm selection) belongs in separate UX: arm analysis and temporal analysis views. Mixing per-context oracle selection into regret conflates two different questions and adds complexity (theta_hat score-space vs reward-space mismatch) without clear value.

---

## Two Reward Signals

Every regret type is computed independently for both:

| Signal | Source | Dense? | Notes |
|--------|--------|--------|-------|
| `immediate` | `event.immediate_reward` | Yes, every event | Automated quality score |
| `human` | `event.human_reward` | Sparse, subset only | Ground truth preference (0/1) |

### Handling Sparsity
- Human windows with fewer than `min_human_events` (default 3) flagged as low-confidence
- Windows with zero human-rated events excluded from human series entirely

### Diagnostic Value of the Gap
- Immediate regret low, human high → automated scoring miscalibrated
- Both low → system working
- Immediate high, human low → optimizing wrong metric

---

## Windowed % Regret

### Formula

For each window `w` of `N` consecutive events:

```
pct_regret_w = sum(optimal_i - chosen_i) / sum(optimal_i) * 100
```

Where:
- `optimal_i` = reward of the oracle's arm for event `i` (per regret type)
- `chosen_i` = reward of the actually-chosen arm for event `i`

### Interpretation
- **0%** = always chose optimal
- **100%** = getting zero when reward was available
- Decreasing curve = Thompson Sampling converging

### Window Size
- Caller-configurable `window_size` parameter
- Default: `max(10, total_events // 20)` → ~20 data points regardless of event count

---

## Oracle Computation Details

### Active Arm Scoping
The oracle is always the best **currently active** arm at the time regret is computed. If an arm is deactivated, regret is recalculated against the next best active arm. This means regret is always measured against a realistic "what you could be doing right now" baseline.

### Accuracy Oracle (best active arm by raw reward)
```python
# Filter to events from active arms only
# Group events by arm_id, compute mean immediate_reward per arm
arm_means = {arm_id: mean(events[arm_id].immediate_reward) for arm_id in active_arm_ids}
best_arm = max(arm_means)

# For each event: optimal = arm_means[best_arm]
# Note: constant per window since oracle is global
```

### Adjusted Oracle (best active arm by adjusted reward)
```python
# Recompute adjusted reward per event from stored fields
adj_reward = calculate_reward(event.immediate_reward, event.cost, event.latency)

# Filter to active arms, group by arm, compute mean adjusted reward per arm
arm_adj_means = {arm_id: mean(adj_rewards[arm_id]) for arm_id in active_arm_ids}
best_arm = max(arm_adj_means)

# For each event: optimal = arm_adj_means[best_arm]
```

---

## Response Schema

```python
class RegretWindow(BaseModel):
    window: int                          # 0-indexed window number
    event_range: tuple[int, int]         # (start_event_id, end_event_id)
    n_events: int                        # events in window

    # Accuracy regret
    accuracy_pct: float                  # % regret
    accuracy_mean_regret: float          # absolute mean gap
    accuracy_mean_optimal: float         # denominator context

    # Adjusted regret
    adjusted_pct: float
    adjusted_mean_regret: float
    adjusted_mean_optimal: float

    # Human variants (nullable when sparse)
    human_accuracy_pct: Optional[float]
    human_adjusted_pct: Optional[float]
    human_n_events: int                  # how many had human feedback
    human_low_confidence: bool           # < min_human_events

class RegretResponse(BaseModel):
    bandit_id: int
    total_events: int
    window_size: int
    windows: list[RegretWindow]

    # Summary
    overall_accuracy_pct: float          # across all events
    overall_adjusted_pct: float

    # Oracle info
    accuracy_best_arm: int               # arm_id
    adjusted_best_arm: int               # arm_id (may differ!)
```

---

## Implementation Plan

### Files to Modify

1. **`app/schemas/bandit_actions.py`** — Add `RegretWindow`, `RegretResponse`, `RegretRequest` schemas
2. **`app/services/bandit/bandit_analysis.py`** — Add `compute_regret()` function with core logic
3. **`app/api/v1/endpoints/bandit_analysis.py`** — Add `GET /{bandit_id}/regret` endpoint

### Implementation Steps

#### Step 1: Schemas (`schemas/bandit_actions.py`)
- `RegretRequest`: `window_size: Optional[int]`, `min_human_events: int = 3`
- `RegretWindow` and `RegretResponse` as above

#### Step 2: Core Logic (`services/bandit/bandit_analysis.py`)

```
compute_regret(session, bandit_id, user_id, window_size, min_human_events):

  1. Load bandit with arms (filter to active)
  2. Load ALL events ordered by created_at

  3. Compute oracles (active arms only):
     a. accuracy_oracle: group events by arm_id → mean(immediate_reward) → best active arm
     b. adjusted_oracle: recompute adj rewards → group by arm_id → mean(adj) → best active arm

  4. For each event, compute two (optimal, chosen) pairs:
     a. accuracy: (accuracy_oracle_mean, event.immediate_reward)
     b. adjusted: (adjusted_oracle_mean, event_adj_reward)

  5. Chunk events into windows of size N

  6. Per window: aggregate into pct_regret
     - Same for human variants (filter to events where human_reward is not None)

  7. Compute overall summary

  8. Return RegretResponse
```

#### Step 3: Endpoint (`api/v1/endpoints/bandit_analysis.py`)
- `GET /{bandit_id}/regret?window_size=30&min_human_events=3`
- Calls `compute_regret()`, returns `RegretResponse`

---

## Resolved Design Questions

1. **Contextualized regret** — Dropped entirely. Regret is an aggregate convergence measure. Per-context analysis (temporal patterns, feature-based arm performance) belongs in separate arm analysis and temporal analysis views.

2. **Warm-up exclusion** — Not needed. Windowed regret already handles this naturally — early high-regret windows are visible but don't contaminate later windows.

3. **Arm lifecycle** — Oracle = best **active** arm at time of computation. Deactivating an arm triggers recalculation against the next best active arm. Regret always measures against a realistic "what you could be doing right now" baseline.
