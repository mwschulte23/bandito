# Cost & Latency Importance

When Bandito scores an arm, it doesn't just look at quality. It also factors in how expensive and how slow the response was. **Importance** controls how much those factors matter.

## The Scale (0-5)

Each bandit has two settings:

- **cost_importance** — How much do you care about API cost?
- **latency_importance** — How much do you care about response time?

| Level | Meaning | Example Use Case |
|-------|---------|-----------------|
| 0 | Don't care at all | Money/speed is irrelevant, only quality matters |
| 1 | Slight preference | Would like cheaper/faster, but barely matters |
| 2 | Moderate (default) | Balanced tradeoff between quality and cost/speed |
| 3 | Important | Willing to sacrifice some quality to save money/time |
| 4 | Very important | Strong penalty for expensive or slow responses |
| 5 | Critical | Near-zero credit for responses that hit the ceiling |

## What It Actually Does

A raw quality score (say 0.85) gets penalized based on cost and latency. Higher importance = steeper penalty.

**Example**: A response scores 0.85 quality and costs $2.50 (half of the $5 max).

| Importance | Adjusted Score | Quality Kept |
|------------|---------------|--------------|
| 0 | 0.85 | 100% |
| 1 | 0.52 | 61% |
| 2 | 0.31 | 37% |
| 3 | 0.19 | 22% |
| 4 | 0.11 | 14% |
| 5 | 0.07 | 8% |

At importance=0, cost is invisible. At importance=5, even moderate cost nearly wipes out the score.

## When to Change It

- **Prototyping / exploring**: Set both to 0 or 1. Focus on finding the best quality arm first.
- **Production with budget**: Set cost_importance to 3-4. The bandit will naturally favor arms that deliver good quality per dollar.
- **Real-time applications**: Set latency_importance to 3-5. Slow responses get penalized hard.
- **Cost-insensitive, latency-sensitive**: cost_importance=0, latency_importance=4. Common for chat applications where speed matters but cost is secondary.

## Defaults

Both default to 2, which matches the balanced tradeoff most experiments start with. You can update them anytime via the bandit update endpoint — the new settings apply to all future reward calculations.
