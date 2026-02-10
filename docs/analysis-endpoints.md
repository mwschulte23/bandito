# Analysis Endpoints

All analysis endpoints live under `/api/v1/bandit/{bandit_id}/` and require authentication.

## Leaderboard

```
GET /{bandit_id}/leaderboard
```

Ranked view of all arms with enriched metrics. Arms are sorted by adjusted human reward (falls back to immediate if no human feedback exists).

Each arm includes:
- **Pull count** and **human feedback count**
- **Reward breakdown** — Raw quality score, cost factor, latency factor, and the final adjusted score. Shows exactly how much each penalty reduced the reward.
- **Bayesian confidence** — Uncertainty estimate from the Thompson Sampling posterior. Lower = more confident.
- **Average cost and latency**

Use this to see which arm is winning and why.

## Regret

```
GET /{bandit_id}/regret?window_size=30&min_human_events=3
```

Windowed regret analysis showing convergence over time. See [Regret Analysis](regret-analysis.md) for full details.

## Forecast

```
GET /{bandit_id}/forecast?hour=14&is_weekend=0&n_simulations=10000&beta=1.0
```

Monte Carlo simulation over the Thompson Sampling posterior. Shows what the bandit would do *right now* for a given context.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hour` | Current hour | Hour of day (0-23) for time-aware features |
| `is_weekend` | Current day | Weekend flag (0 or 1) |
| `n_simulations` | 10,000 | Number of Monte Carlo draws (100-100k) |
| `beta` | 1.0 | Exploration parameter. Higher = more random. |

Returns per-arm:
- **Selection probability** — How often this arm would be picked
- **Expected score** — Mean predicted reward
- **Uncertainty** — Standard deviation across simulations

Includes a human-readable confidence note based on uncertainty levels.

## State Analysis

```
GET /{bandit_id}/analysis
```

Deep dive into the bandit's internal learned weights. Decomposes the parameter vector (theta) into interpretable components:

- **Feature weights** — How much each feature (model type, prompt style, time of day) influences arm selection
- **Uncertainty per feature** — Which dimensions the bandit is still uncertain about
- **Model ranking** — Arms ranked by their baseline effect (intercept)

Useful for understanding *why* the bandit prefers certain arms, not just *which* ones it prefers.

## Budget

```
GET /{bandit_id}/budget
```

Current spend tracking against the bandit's budget limit.

Returns:
- **total_spent** — Sum of all arm pull costs
- **budget** — The configured limit (null if no budget)
- **remaining** — How much is left
- **used_percent** — Percentage consumed (0-100)
- **is_over_budget** — Boolean flag
