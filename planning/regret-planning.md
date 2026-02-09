# Regret Analysis — Implementation Plan

## Overview

Add regret computation to `bandit_analysis.py` exposing three layered regret types, each computed for both reward signals (immediate + human), returned as a windowed time series of % regret.

---

## Three Regret Types

### 1. Accuracy Regret
- **Question answered**: "Are we converging on the highest quality arm?"
- **Reward signal**: `immediate_reward` (raw quality) or `human_reward` (raw binary)
- **Oracle**: Global best arm — the arm with the highest mean raw reward across all events
- **No cost/latency adjustment**

### 2. Adjusted Regret
- **Question answered**: "Are we converging on the best *value* arm?"
- **Reward signal**: `calculate_reward(immediate_reward, cost, latency)` — recomputed from stored fields
- **Oracle**: Global best arm by mean adjusted reward
- **Captures cost/latency tradeoffs** — a slightly worse arm that's cheaper/faster may be optimal here

### 3. Contextualized Regret
- **Question answered**: "Are we picking the right arm *for this moment*?"
- **Reward signal**: Adjusted reward (same as above)
- **Oracle**: Per-event best arm via `theta_hat @ features` for each event's context
- **Uses `FeatureTransformer`** to rebuild feature vectors for all arms against each event's context
- **This is the only regret type that can reveal temporal patterns** (arm A wins weekdays, arm B wins weekends)

### Layer Progression

```
Accuracy:      raw reward    →  global best arm
Adjusted:      adj reward    →  global best arm
Contextualized: adj reward   →  per-context best arm (via theta_hat)
```

Each layer adds a dimension. Divergence between layers is diagnostic:
- Accuracy low, Adjusted high → best quality arm is expensive/slow, system correctly trading off
- Adjusted low, Contextualized high → system found the right arm overall but misses temporal patterns
- All low → converged, Thompson Sampling is working

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

### Accuracy Oracle (global best arm by raw reward)
```python
# Group events by arm_id, compute mean immediate_reward per arm
arm_means = {arm_id: mean(events[arm_id].immediate_reward)}
best_arm = max(arm_means)

# For each event: optimal = arm_means[best_arm]
# Note: constant per window since oracle is global
```

### Adjusted Oracle (global best arm by adjusted reward)
```python
# Recompute adjusted reward per event from stored fields
adj_reward = calculate_reward(event.immediate_reward, event.cost, event.latency)

# Group by arm, compute mean adjusted reward per arm
arm_adj_means = {arm_id: mean(adj_rewards[arm_id])}
best_arm = max(arm_adj_means)

# For each event: optimal = arm_adj_means[best_arm]
```

### Contextualized Oracle (per-event best arm via theta_hat)
```python
# For each event:
mapper = FeatureTransformer(arms)
for arm in active_arms:
    features = mapper.transform_to_vector(arm, event.context)
    score = features @ theta_hat
best_arm_for_context = argmax(scores)

# optimal = calculate_reward(???)
# Problem: we don't know what reward the oracle arm *would have gotten*
# Solution: use theta_hat score as the expected reward proxy
# For chosen arm: also use theta_hat score (not actual reward)
# This gives regret in score-space, which is what theta_hat optimizes
```

**Important**: Contextualized regret lives in score-space (theta_hat projections), not reward-space. This is correct — it measures regret against the policy Thompson Sampling is *trying* to learn, not against noisy realized rewards.

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

    # Contextualized regret
    contextualized_pct: float
    contextualized_mean_regret: float
    contextualized_mean_optimal: float

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
    overall_contextualized_pct: float

    # Oracle info
    accuracy_best_arm: int               # arm_id
    adjusted_best_arm: int               # arm_id (may differ!)
    contextualized_note: str             # "per-context, no single best arm"
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

  1. Load bandit with arms + state (theta_hat needed for contextualized)
  2. Load ALL events ordered by created_at
  3. Build FeatureTransformer from arms

  4. Compute oracles:
     a. accuracy_oracle: group events by arm_id → mean(immediate_reward) → best arm
     b. adjusted_oracle: recompute adj rewards → group by arm_id → mean(adj) → best arm
     c. contextualized: precompute theta_hat scores per event (all arms)

  5. For each event, compute three (optimal, chosen) pairs:
     a. accuracy: (accuracy_oracle_mean, event.immediate_reward)
     b. adjusted: (adjusted_oracle_mean, event_adj_reward)
     c. contextualized: (max_score_all_arms, chosen_arm_score) via theta_hat

  6. Chunk events into windows of size N

  7. Per window: aggregate into pct_regret
     - Same for human variants (filter to events where human_reward is not None)

  8. Compute overall summary

  9. Return RegretResponse
```

#### Step 3: Endpoint (`api/v1/endpoints/bandit_analysis.py`)
- `GET /{bandit_id}/regret?window_size=30&min_human_events=3`
- Calls `compute_regret()`, returns `RegretResponse`

---

## Open Questions

1. **Contextualized human regret** — human_reward is binary (0/1) and never adjusted. Should contextualized human regret use theta_hat score-space (consistent with contextualized immediate) or skip it entirely since human signal doesn't go through the same scoring?

2. **Warm-up exclusion** — Should early events (first N, or first K per arm) be excludable? Early exploration is *expected* to have high regret. Could add `exclude_first_n: int = 0` parameter.

3. **Arm lifecycle** — If arms are added/deactivated mid-experiment, the global oracle changes. Should we recompute the oracle only over arms that were active at each event's timestamp, or use the final arm set?
