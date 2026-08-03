import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv("POSTGRES_DB_URL", "")

# Convert postgresql:// to postgresql+psycopg:// for async SQLAlchemy with psycopg v3
if DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL

# Create the async engine
engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)

# Async session factory
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
