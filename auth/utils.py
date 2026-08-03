import os
import re
import datetime
import hashlib
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-default-development-jwt-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30

ph = PasswordHasher()

def hash_password(password: str) -> str:
    """Hash password using Argon2."""
    return ph.hash(password)

def verify_password(hashed_password: str, password: str) -> bool:
    """Verify an Argon2 password hash against a password."""
    try:
        ph.verify(hashed_password, password)
        return True
    except VerifyMismatchError:
        return False

def validate_password_strength(password: str) -> None:
    """Validate that password meets minimum complexity requirements."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one digit")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise ValueError("Password must contain at least one special character")

def create_access_token(user_id: str, email: str) -> str:
    """Generate JWT Access Token with 15-minute expiry."""
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": int(expire.timestamp())
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    """Generate JWT Refresh Token with 30-day expiry."""
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "user_id": user_id,
        "exp": int(expire.timestamp())
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt_token(token: str) -> dict | None:
    """Decode and verify a JWT token. Returns payload if valid, None otherwise."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def hash_token(token: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
