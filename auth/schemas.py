import re
import datetime
from uuid import UUID
from pydantic import BaseModel, Field, field_validator
from auth.utils import validate_password_strength

class RegisterRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
    name: str | None = Field(None, description="Optional user name")
    age: int | None = Field(None, description="Optional user age")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        value = value.strip().lower()
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, value):
            raise ValueError("Invalid email address format")
        return value

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, value: str) -> str:
        validate_password_strength(value)
        return value

class LoginRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return value.strip().lower()

class UserResponse(BaseModel):
    id: UUID
    email: str
    name: str | None = None
    age: int | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    access_token: str | None = None

    class Config:
        from_attributes = True

class MessageResponse(BaseModel):
    status: str
    message: str
