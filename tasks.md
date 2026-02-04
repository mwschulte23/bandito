# Bandito - Improvement Tasks

Prioritized list of codebase improvements identified during architectural review.

---

## High Priority

### 1. Create unified BanditService class

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

---

## Bugs

### 1. Investigate cost/latency not written to events

Despite appearing to be present in the request, cost and latency values aren't being persisted to events.

**To investigate:**
- Check `update_on_reward()` in `bandit_brain.py` - are values being set on event?
- Check `update_event()` in `data_helper.py` - are fields being copied correctly?
- Verify database column types match expected values
- Add logging to trace values through the flow

---

## Testing

### 2. Add tests/QA scripts for refactored codebase

After significant refactoring (session management, schema renames, endpoint changes), need test coverage to prevent regressions.

**Scope:**
- Unit tests for service layer (`bandit_brain.py`, `helpers.py`)
- Integration tests for action endpoints (pull, reward, feedback)
- SDK integration test script
- Consider pytest fixtures for test database

---

## Feature Work

### 3. Add segments to events via SDK/API

**Gap:** Currently no way to attach segments to events through the action endpoints or SDK.

**Current state:**
- `EventSegment` model exists
- `POST /{bandit_id}/events` accepts segments (CRUD endpoint)
- But `pull_arm` + `reward` flow doesn't capture segments

**Options:**
1. Add `segments` param to `pull()` - attach at pull time
2. Add `segments` param to `reward()` - attach with reward submission
3. Separate endpoint `POST /{bandit_id}/events/{event_id}/segments`

**Considerations:**
- When is segment info known? (likely at pull time from request context)
- SDK needs corresponding method

---

### 4. Segment analysis

**Source:** CLAUDE.md TODO section

**Description:**
- Split bandits by segment (e.g., `bandit_name` → `bandit_name_mobile` + `bandit_name_desktop`)
- "What-if" analysis: view bandit state for event subsets

**Considerations:**
- Low-frequency, potentially expensive operations
- May need separate router from hot-path pull operations
- Consider caching strategies

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
