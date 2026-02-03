from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import HTTPException

from app.models.user import User
from app.core.security import verify_password, get_password_hash
from app.core.config import settings


class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id: int) -> Optional[User]:
        statement = select(User).where(User.id == id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        statement = select(User).where(User.email == email)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        user = await self.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user

    async def create(self, email: str, password: str, is_active: bool = True) -> User:
        is_superuser = email in settings.ALLOWED_SUPERUSER_EMAILS
        user = User(
            email=email,
            hashed_password=get_password_hash(password),
            is_superuser=is_superuser,
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def update_password(self, user_id: int, new_password: str):
        user = await self.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        hashed_password = get_password_hash(new_password)
        user.hashed_password = hashed_password
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def update_status(self, user_id: int, is_active: bool) -> User:
        user = await self.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user.is_active = is_active
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def delete(self, user_id: int) -> User:
        user = await self.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        await self.session.delete(user)
        await self.session.commit()
        return user