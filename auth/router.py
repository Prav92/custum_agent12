import logging
import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.session import get_db
from db.models import User, RefreshToken
from auth.schemas import RegisterRequest, LoginRequest, UserResponse, MessageResponse
from auth.utils import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_token,
    verify_jwt_token,
)
from auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

logger = logging.getLogger("auth_router")

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalars().first()
    if existing_user:
        logger.warning(f"Registration failed: email {request.email} already registered")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_pwd = hash_password(request.password)
    user = User(
        email=request.email,
        password_hash=hashed_pwd,
        name=request.name,
        age=request.age
    )
    
    db.add(user)
    try:
        await db.commit()
        await db.refresh(user)
        logger.info(f"User {user.email} successfully registered (ID: {user.id})")
        return user
    except Exception as e:
        await db.rollback()
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not register user"
        )

@router.post("/login", response_model=UserResponse)
async def login(request: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Authenticate user, create tokens, set HttpOnly cookies, and return user profile."""
    # Fetch user
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalars().first()
    
    # Verify password
    if not user or not verify_password(user.password_hash, request.password):
        logger.warning(f"Login failed: invalid credentials for email {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate tokens
    access_token = create_access_token(str(user.id), user.email)
    refresh_token = create_refresh_token(str(user.id))
    
    # Save hashed refresh token to database
    token_hash = hash_token(refresh_token)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    db_token = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at
    )
    
    db.add(db_token)
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to save refresh token for user {user.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to database error"
        )

    # Set cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=15 * 60,  # 15 minutes
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=30 * 24 * 60 * 60,  # 30 days
        path="/"
    )
    
    logger.info(f"User {user.email} successfully logged in")
    return user

@router.post("/refresh", response_model=MessageResponse)
async def refresh(response: Response, refresh_token: str | None = Cookie(default=None), db: AsyncSession = Depends(get_db)):
    """Verify refresh token, delete/revoke old one, issue new pair of tokens and rotate."""
    if not refresh_token:
        logger.warning("Refresh failed: refresh token cookie is missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing"
        )
    
    # Decode and verify JWT structure/expiry
    payload = verify_jwt_token(refresh_token)
    if not payload:
        logger.warning("Refresh failed: invalid or expired refresh token JWT signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Fetch from DB using hash
    hashed_old_token = hash_token(refresh_token)
    result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == hashed_old_token))
    db_token = result.scalars().first()
    
    if not db_token or db_token.revoked or db_token.expires_at < datetime.datetime.now(datetime.timezone.utc):
        logger.warning("Refresh failed: token hash not found, already revoked, or expired in database")
        # Security: if token exists but is invalid, we delete/revoke it to clean up
        if db_token:
            await db.delete(db_token)
            await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token"
        )
    
    # Get user details for new access token
    user_result = await db.execute(select(User).where(User.id == db_token.user_id))
    user = user_result.scalars().first()
    if not user:
        # User no longer exists, clean up token and reject
        await db.delete(db_token)
        await db.commit()
        logger.warning("Refresh failed: user associated with token no longer exists")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Generate new tokens
    new_access_token = create_access_token(str(user.id), user.email)
    new_refresh_token = create_refresh_token(str(user.id))
    
    # Delete old refresh token from DB (rotation)
    await db.delete(db_token)
    
    # Save new refresh token hash to DB
    new_token_hash = hash_token(new_refresh_token)
    new_expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    db_new_token = RefreshToken(
        user_id=user.id,
        token_hash=new_token_hash,
        expires_at=new_expires_at
    )
    db.add(db_new_token)
    
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Rotation commit failed for user {user.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Refresh failed due to database error"
        )
    
    # Set updated cookies
    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=15 * 60,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=30 * 24 * 60 * 60,
        path="/"
    )
    
    logger.info(f"Successfully rotated tokens for user {user.email}")
    return MessageResponse(status="success", message="Tokens refreshed successfully")

@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response, refresh_token: str | None = Cookie(default=None), db: AsyncSession = Depends(get_db)):
    """Remove refresh token from database, clear client-side cookies, and return success."""
    if refresh_token:
        hashed_token = hash_token(refresh_token)
        result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == hashed_token))
        db_token = result.scalars().first()
        if db_token:
            await db.delete(db_token)
            try:
                await db.commit()
                logger.info(f"Revoked refresh token from database on logout")
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to delete refresh token on logout: {str(e)}")
    
    # Clear cookies
    response.delete_cookie(key="access_token", path="/", httponly=True, secure=True, samesite="lax")
    response.delete_cookie(key="refresh_token", path="/", httponly=True, secure=True, samesite="lax")
    
    return MessageResponse(status="success", message="Logged out successfully")

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve details of the currently authenticated user."""
    return current_user
