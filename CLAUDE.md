# Bandito

Multi-armed bandit optimization API for LLM/AI agents. Each "step" pulls an arm (model + prompt configuration) and over time each arm is scored using Thompson Sampling.

## Tech Stack

- FastAPI (async)
- PostgreSQL with async SQLAlchemy/SQLModel
- Alembic migrations
- NumPy for bandit math
- JWT auth with bcrypt

## Key Concepts

- **Bandit**: Container for an experiment/optimization
- **Arm**: A model + prompt configuration to test
- **State**: NumPy arrays (A, b, theta, cholesky) stored as binary blobs, enabling stateless reconstruction
- **Event**: Record of an arm pull with feature vector and reward signals
- **Segment**: User context (device, region, etc.) attached to events for post-hoc analysis

## Project Structure

```
app/
├── api/v1/endpoints/     # Route handlers
├── core/                 # Config, security, deps
├── models/               # SQLModel ORM models
├── schemas/              # Pydantic request/response
├── services/bandit/      # Algorithm implementation (WIP)
└── main.py
```

## Running

```bash
uvicorn app.main:app --reload
```

## Testing

```bash
python scratch/test_endpoints.py
```

---

## TODO: Router Architecture for Actions

Currently have `bandit.py` for CRUD. Need `bandit_actions.py` for core operations.

### Planned Actions

1. **pull_arm** - Thompson Sampling arm selection (hot path, low latency)
2. **reward** - Apply reward to an event

### Future: Segment Analysis

- Split bandits by segment (e.g., `bandit_name` → `bandit_name_mobile` + `bandit_name_desktop`)
- "What-if" analysis: view bandit state for event subsets

### Architecture Decision

Start with two routers:
- `bandit.py` - CRUD (setup/admin)
- `bandit_actions.py` - Runtime operations

Consider splitting to three if segment analysis grows:
- `bandit.py` - CRUD
- `pulls.py` - pull_arm, reward (hot path)
- `analysis.py` - segment splits, what-if scenarios

**Rationale**: Pull arm is high-frequency/low-latency. Segment analysis is low-frequency/potentially expensive. Different operational characteristics may need separation for caching, rate limiting, optimization.
