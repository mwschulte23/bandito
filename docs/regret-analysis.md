# Regret Analysis

Regret answers one question: **is the bandit learning?**

It measures how much reward you "left on the table" compared to always picking the best arm. A decreasing regret trend means Thompson Sampling is converging — the system is figuring out which arm wins.

## The Endpoint

```
GET /api/v1/bandit/{bandit_id}/regret?window_size=30&min_human_events=3
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | ~20 windows auto-calculated | Events per window. Smaller = more granular, noisier. |
| `min_human_events` | 3 | Minimum human-rated events in a window before it's considered confident. |

## What You Get Back

The response has two layers: **per-window trends** and an **overall summary**.

### Four Regret Metrics

Each window reports four metrics, and each metric has its own oracle (best arm):

| Metric | What It Measures | Oracle |
|--------|-----------------|--------|
| **accuracy** | Raw quality only | Best arm by mean `immediate_reward` |
| **adjusted** | Quality penalized by cost & latency | Best arm by mean adjusted reward |
| **cost** | Cost penalty in isolation | Lowest-cost arm |
| **latency** | Latency penalty in isolation | Fastest arm |

Each metric includes:
- **pct** — % of optimal reward left on the table (0% = perfect, 100% = getting nothing)
- **mean_regret** — Average per-event gap between oracle and chosen arm
- **mean_optimal** — The oracle's mean reward (context for the numbers)
- **best_arm_id** — Which arm is the oracle for this metric

### Human Variants

If you're collecting human feedback (`human_reward`), you also get `human_accuracy` and `human_adjusted` metrics. These use human scores instead of immediate rewards. Since human feedback is usually sparse, windows with fewer than `min_human_events` human-rated events are flagged as `human_low_confidence: true`.

## Reading the Results

### The Convergence Curve

Plot `accuracy.pct` and `adjusted.pct` across windows. You want to see a **downward trend**.

```
Window 0:  accuracy 45%  adjusted 52%   ← exploring, high regret
Window 5:  accuracy 22%  adjusted 28%   ← learning
Window 10: accuracy 8%   adjusted 11%   ← converging
Window 15: accuracy 3%   adjusted 5%    ← near-optimal
```

### Diagnostic: Accuracy vs Adjusted

The gap between accuracy and adjusted regret tells you something:

- **Both low** — The system found the best arm and it's also efficient. You're done.
- **Accuracy low, adjusted high** — The best-quality arm is expensive or slow. The bandit is correctly trading quality for efficiency.
- **Accuracy high, adjusted low** — Unlikely, but means you're picking cheap/fast arms that aren't the best quality.

### Diagnostic: Immediate vs Human

If you have both signals:

- **Immediate low, human high** — Your automated scoring doesn't match human judgment. Recalibrate your immediate reward.
- **Both low** — Automated and human scoring agree. The system is working.
- **Immediate high, human low** — You're optimizing the wrong metric. Human feedback says the arm the bandit likes isn't actually best.

## How the Oracle Works

The oracle is the best **active** arm based on all events across the entire run. Each metric picks its own oracle independently — the best-quality arm and the most cost-efficient arm might be different.

If you deactivate an arm, regret recalculates against the next best active arm. The oracle always represents "what you could be doing right now."

## Example Response (Abbreviated)

```json
{
  "bandit_id": 42,
  "total_events": 150,
  "window_size": 30,
  "overall": {
    "accuracy": {"pct": 12.3, "mean_regret": 0.08, "mean_optimal": 0.65, "best_arm_id": 7},
    "adjusted": {"pct": 18.1, "mean_regret": 0.06, "mean_optimal": 0.33, "best_arm_id": 3}
  },
  "windows": [
    {
      "window": 0,
      "n_events": 30,
      "metrics": {
        "accuracy": {"pct": 35.2, "mean_regret": 0.23, "mean_optimal": 0.65, "best_arm_id": 7},
        "adjusted": {"pct": 41.0, "mean_regret": 0.14, "mean_optimal": 0.33, "best_arm_id": 3}
      },
      "human_metrics": {"accuracy": null, "adjusted": null},
      "human_n_events": 1,
      "human_low_confidence": true
    }
  ]
}
```
