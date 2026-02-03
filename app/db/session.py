import asyncio
from sqlmodel import SQLModel
from typing import AsyncGenerator
from sqlalchemy import text
# from sqlalchemy.pool import NullPool #used
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    echo=settings.DB_ECHO,
    future=True,
    pool_pre_ping=True,  # Enables connection health checks
    pool_size=settings.POOL_SIZE,
    max_overflow=settings.MAX_OVERFLOW,
    pool_timeout=settings.POOL_TIMEOUT,
    pool_recycle=settings.POOL_RECYCLE,
    # poolclass=NullPool  # Disable pooling
) # not using connection pooling for now, causes celery task issues...

async def init_db():
    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all)  # Uncomment to reset db
        await conn.run_sync(SQLModel.metadata.create_all)


async def _test_connection(session: AsyncSession) -> bool:
    """Test if the database connection is valid."""
    try:
        await session.execute(text("SELECT 1"))
        return True
    except OperationalError as e:
        return False



async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    retry_count = 0
    session = async_session()

    while retry_count < settings.MAX_RETRIES:
        try:
            if await _test_connection(session):
                break
            
            retry_count += 1
            if retry_count == settings.MAX_RETRIES:
                raise OperationalError("Failed to establish database connection after maximum retries")
            
            await session.close()
            await asyncio.sleep(settings.RETRY_DELAY)
            session = async_session()
            
        except Exception as e:
            # logger.error(f"Error during connection attempt {retry_count}: {str(e)}")
            await session.close()
            raise

    try:
        yield session
    finally:
        await session.close()


if __name__ == '__main__':
    asyncio.run(init_db())
    