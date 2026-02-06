# Bandito - Tasks

Prioritized list of codebase improvements identified during architectural review.

---

### Handle case when user changes human reward from 0 -> 1 or vice versa
This will require handling bandit weights appropriately as well as updating the PATCH event/{event_id} endpoint


### Add filter on the list events endpoint

The filters I'd like to *consider* adding are
1. Return events without a human reward
2. Return events without an immediate award
3. Return events without any reward
4. Anything else???

### Budget endpoint math is unclear

The budget used percent range is 0-100, don't do that transform. 0.74 should equal 74%
```
{
  "bandit_id": 1,
  "budget": 1,
  "current_spend": 0.0074,
  "budget_remaining": 0.9926,
  "budget_used_percent": 0.74,
  "is_over_budget": false
}
```

### Create unified BanditService class

**Current state:** Business logic scattered across:
- `app/services/bandit/bandit_brain.py`
- `app/services/bandit/bandit_analysis.py`
- `app/services/bandit/utils/data_helper.py`

**Fix:** Create cohesive `BanditService` class with injected session:

```python
class BanditService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def pull_arm(self, bandit_id: int, ...) -> dict
    async def apply_reward(self, event_id: int, ...) -> None
    async def get_budget_status(self, bandit_id: int) -> dict
    # etc.
```

**Benefits:**
- Improved testability
- Proper transaction boundaries
- Centralized business logic


### Create unified BanditService class

Add suite of tests.
- Creation -> does bandit, arm and state work as expected
- Pull/update -> does math work properly, are state and events responses and objects CRUD'd properly
- Analytics -> SKIP FOR NOW
- SDK -> does SDK properly handle endpoints used? is it up-to-date w/ API?

Primary purpose: Allows me to give more trust to AI coding tools, e.g claude code. Especially in doing more extensive work.

### FEATURE: Implement segments to events in SDK & API

There is a segment object that enables a user to add context like user device. These segments are then useful for
1. Analytics: Overall Arm #1 is great, but inferior on mobile.
2. Bandit branching: Puts a user action behind analytics case above. If mobile stinks on best arm, "branch" bandit.
    * Bandit branching will require deep technical discussion to properly handle. Impacts huge chunk of codebase
    * E.g a bandit family that handles branches? or a totally different bandit? Impact on codebase and user experience

**Current state:**
- `EventSegment` model exists
- `POST /{bandit_id}/events` accepts segments (CRUD endpoint)
- But `pull_arm` + `reward` flow doesn't capture segments

### DISCUSSION: Breakdown interface for python SDK...pros/cons, most popular existing patterns in python / AI space.
- Discuss implementation alternatives with claude code.
    - A decorator or context manager path for arm / update
    - A built-in streamlit dashboard for analytics??

---

## For Future:

### DISCUSSION: Discuss building JS/TS SDK

Eventually, an SDK for frontend devs will be important!


---

## Completed

- [x] Extract duplicate `get_bandit_for_user` to shared helpers module
- [x] Extract duplicate dimension calculation to `calculate_bandit_dimensions()`
- [x] Extract duplicate budget calculation to `calculate_budget_status()`
- [x] Fix missing HTTPException import in `data_helper.py`
- [x] Fix generic exception swallowing in `bandit.py` (removed unnecessary try/except)
- [x] Standardize session management in `data_helper.py` and `bandit_brain.py`
- [x] Remove dead code (scratch directory, duplicate numpy import)
- [x] Remove PATCH events endpoint (bypassed bandit algorithm) + BanditEventUpdate schema
- [x] Schema naming convention standardization (Hybrid: CRUD=*Read/*Create/*Update, Actions=*Request/*Response)
- [x] Fix 404 on reward endpoint (missing `session.commit()` after session refactor)
