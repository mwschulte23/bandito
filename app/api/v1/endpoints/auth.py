from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
# from sqlmodel import Session
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.security import create_access_token, create_reset_token, verify_reset_token
from app.core.config import settings
from app.db.session import get_session
from app.services.user_service import UserService
# from app.services.email.emailer import send_password_reset_email
from app.schemas.token import Token, PasswordReset
from app.core.deps import get_current_user


router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    session: AsyncSession = Depends(get_session),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = await UserService(session).authenticate(email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    # if not user.is_active:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    # if not user.is_subscribed:
    #     raise HTTPException(status_code=303, detail="You need to setup subscription to use this app")
    
    return {
        "access_token": create_access_token(str(user.id)),
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
    }

@router.post("/signup", response_model=Token)
async def signup(
    session: AsyncSession = Depends(get_session),
    email: str = Form(...),
    password: str = Form(...)
):
    user = await UserService(session).get_by_email(email)
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
       # delete users who signup, but don't subscribe. enables signing up again w/ same email
    #    if not user.subscription_id:
    #        await UserService(session).delete(user.id)
    #    else:
    #        raise HTTPException(status_code=400, detail="Email already registered")
   
    user = await UserService(session).create(email=email, password=password)
    return {
        "access_token": create_access_token(str(user.id)),
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
    }

@router.patch("/{user_id}/status")
async def update_user_status(
    user_id: int,
    is_active: bool,
    current_user = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
   if not current_user.is_superuser:
       raise HTTPException(status_code=403, detail="Not enough permissions")
   return await UserService(session).update_status(user_id, is_active)


@router.get("/me")
async def get_current_user_info(
   current_user = Depends(get_current_user)
):
   return current_user


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    return await UserService(session).delete(user_id)


@router.post("/reset-password-request/")
async def request_password_reset(
    email: str,
    session: AsyncSession = Depends(get_session)
):
    user = await UserService(session).get_by_email(email)
    if user:
        reset_token = create_reset_token(str(user.id))
        # send_password_reset_email(user.email, reset_token)
        success = True
    else:
        success = False
    return {"success": success}


@router.post("/reset-password/")
async def reset_password(
    password_reset: PasswordReset,
    session: AsyncSession = Depends(get_session)
):
    user_id = verify_reset_token(password_reset.token)
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired reset token"
        )
    await UserService(session).update_password(int(user_id), password_reset.new_password)
    return {"message": "Password updated successfully"}