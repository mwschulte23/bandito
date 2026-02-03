from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.core.deps import verify_user
from app.core.deps import get_current_user
from app.models.user import User
from app.models.bandit import Bandit, BanditArm, BanditState, BanditEvent, EventSegment
from app.schemas.bandit import (
    BanditCreate, BanditRead, BanditUpdate, BanditReadWithArms,
    BanditArmCreate, BanditArmRead, BanditArmUpdate,
    BanditStateResponse,
    BanditEventUpdate, BanditEventCreateWithSegments, BanditEventResponse,
    EventSegmentRead,
)


router = APIRouter(dependencies=[Depends(verify_user)])

# ============ Helper Functions ============

async def get_bandit_for_user(
    session: AsyncSession, bandit_id: int, user_id: int
) -> Bandit | None:
    """Helper to fetch a bandit with ownership verification."""
    result = await session.execute(
        select(Bandit).where(Bandit.id == bandit_id, Bandit.user_id == user_id)
    )
    return result.scalar_one_or_none()


# ============ Bandit CRUD ============

@router.post("/", response_model=BanditRead, status_code=status.HTTP_201_CREATED)
async def create_bandit(
    bandit_in: BanditCreate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    try:
        bandit = Bandit(**bandit_in.model_dump(), user_id=current_user.id)
        session.add(bandit)
        await session.flush()
        state = BanditState.create_new(bandit_id=bandit.id, dimensions=0)
        session.add(state)

        await session.commit()
        await session.refresh(bandit)
        return bandit
    except Exception as e:
        raise HTTPException(status_code=400, detail="Issue creating bandit")


@router.get("/", response_model=List[BanditRead])
async def list_bandits(
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    result = await session.execute(
        select(Bandit).where(Bandit.user_id == current_user.id) \
            .order_by(Bandit.updated_at.desc()).limit(limit).offset(skip)
    )
    return result.scalars().all()


@router.get("/{bandit_id}", response_model=BanditReadWithArms)
async def get_bandit(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    result = await session.execute(
        select(Bandit)
        .where(Bandit.id == bandit_id, Bandit.user_id == current_user.id)
        .options(selectinload(Bandit.arms))
    )
    bandit = result.scalar_one_or_none()
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")
    return bandit


@router.patch("/{bandit_id}", response_model=BanditRead)
async def update_bandit(
    bandit_id: int,
    bandit_in: BanditUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    for key, value in bandit_in.model_dump(exclude_unset=True).items():
        setattr(bandit, key, value)

    session.add(bandit)
    await session.commit()
    await session.refresh(bandit)
    return bandit


@router.delete("/{bandit_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bandit(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    await session.delete(bandit)
    await session.commit()
    return None


# ============ BanditArm CRUD ============

@router.post("/{bandit_id}/arms", response_model=BanditArmRead, status_code=status.HTTP_201_CREATED, tags=['bandit_arm'])
async def create_arm(
    bandit_id: int,
    arm_in: BanditArmCreate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    arm = BanditArm(**arm_in.model_dump(), bandit_id=bandit_id)
    session.add(arm)
    

    # PROBZ MOVE THIS ELSEWHERE
    results = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id)
    )
    all_arms = results.scalars().all()
    models = set(arm.model_name for arm in all_arms)
    prompts = set(arm.system_prompt for arm in all_arms)
    new_dimensions = len(models) + len(prompts) + (len(models) * 3) 

    res = await session.execute(select(BanditState).where(BanditState.bandit_id == bandit_id))
    state = res.scalar_one_or_none()
    state.resize(new_dimensions=new_dimensions)
    session.add(state)

    await session.commit()
    await session.refresh(arm)

    return arm


@router.get("/{bandit_id}/arms", response_model=List[BanditArmRead], tags=['bandit_arm'])
async def list_arms(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id)
    )
    return result.scalars().all()


@router.get("/{bandit_id}/arms/{arm_id}", response_model=BanditArmRead, tags=['bandit_arm'])
async def get_arm(
    bandit_id: int,
    arm_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id, BanditArm.id == arm_id)
    )
    arm = result.scalar_one_or_none()
    if not arm:
        raise HTTPException(status_code=404, detail="Arm not found")
    return arm


@router.patch("/{bandit_id}/arms/{arm_id}", response_model=BanditArmRead, tags=['bandit_arm'])
async def update_arm(
    bandit_id: int,
    arm_id: int,
    arm_in: BanditArmUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id, BanditArm.id == arm_id)
    )
    arm = result.scalar_one_or_none()
    if not arm:
        raise HTTPException(status_code=404, detail="Arm not found")

    for key, value in arm_in.model_dump(exclude_unset=True).items():
        setattr(arm, key, value)

    session.add(arm)
    await session.commit()
    await session.refresh(arm)

    return arm


@router.delete("/{bandit_id}/arms/{arm_id}", status_code=status.HTTP_204_NO_CONTENT, tags=['bandit_arm'])
async def delete_arm(
    bandit_id: int,
    arm_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id, BanditArm.id == arm_id)
    )
    arm = result.scalar_one_or_none()
    if not arm:
        raise HTTPException(status_code=404, detail="Arm not found")

    await session.delete(arm)
    await session.commit()

    return None


# ============ BanditState CRU ============

@router.post("/{bandit_id}/state", response_model=BanditStateResponse, status_code=status.HTTP_201_CREATED)
async def create_state(
    bandit_id: int,
    dimensions: int = 17,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    # Check if state already exists
    existing = await session.execute(
        select(BanditState).where(BanditState.bandit_id == bandit_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="State already exists for this bandit")

    state = BanditState.create_new(bandit_id=bandit_id, dimensions=dimensions)
    session.add(state)
    await session.commit()
    await session.refresh(state)
    return state


@router.get("/{bandit_id}/state", response_model=BanditStateResponse)
async def get_state(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditState).where(BanditState.bandit_id == bandit_id)
    )
    state = result.scalar_one_or_none()
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    return state


# ============ BanditEvent CRUD ============

@router.post("/{bandit_id}/events", response_model=BanditEventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    bandit_id: int,
    event_in: BanditEventCreateWithSegments,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    # Verify arm exists and belongs to this bandit
    arm_result = await session.execute(
        select(BanditArm).where(
            BanditArm.id == event_in.arm_id,
            BanditArm.bandit_id == bandit_id
        )
    )
    if not arm_result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Arm not found for this bandit")

    # Create the event (exclude segments from dump)
    event_data = event_in.model_dump(exclude={'segments'})
    event = BanditEvent(
        **event_data,
        bandit_id=bandit_id
    )
    session.add(event)
    await session.flush()  # Get the event ID for segments

    # Create segments
    for seg in event_in.segments:
        segment = EventSegment(
            bandit_event_id=event.id,
            segment_name=seg.get('segment_name', ''),
            segment_value=seg.get('segment_value', '')
        )
        session.add(segment)

    await session.commit()
    await session.refresh(event)
    return event


@router.get("/{bandit_id}/events", response_model=List[BanditEventResponse])
async def list_events(
    bandit_id: int,
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditEvent)
        .where(BanditEvent.bandit_id == bandit_id)
        .order_by(BanditEvent.created_at.desc())
        .limit(limit)
        .offset(skip)
    )
    return result.scalars().all()


@router.get("/{bandit_id}/events/{event_id}", response_model=BanditEventResponse)
async def get_event(
    bandit_id: int,
    event_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditEvent).where(
            BanditEvent.bandit_id == bandit_id,
            BanditEvent.id == event_id
        )
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.patch("/{bandit_id}/events/{event_id}", response_model=BanditEventResponse)
async def update_event(
    bandit_id: int,
    event_id: int,
    event_in: BanditEventUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    result = await session.execute(
        select(BanditEvent).where(
            BanditEvent.bandit_id == bandit_id,
            BanditEvent.id == event_id
        )
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    for key, value in event_in.model_dump(exclude_unset=True).items():
        setattr(event, key, value)

    session.add(event)
    await session.commit()
    await session.refresh(event)
    return event


# ============ EventSegment Read ============

@router.get("/{bandit_id}/events/{event_id}/segments", response_model=List[EventSegmentRead])
async def list_segments(
    bandit_id: int,
    event_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify bandit ownership
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    # Verify event exists for this bandit
    event_result = await session.execute(
        select(BanditEvent).where(
            BanditEvent.bandit_id == bandit_id,
            BanditEvent.id == event_id
        )
    )
    if not event_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Event not found")

    result = await session.execute(
        select(EventSegment).where(EventSegment.bandit_event_id == event_id)
    )
    return result.scalars().all()
