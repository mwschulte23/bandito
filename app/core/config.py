import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    # General
    PROJECT_NAME: str = 'Bandits'
    ALLOWED_ORIGINS: List[str] = ["localhost:5173", "*"]
    API_V1_STR: str = "/api/v1"
    # STRIPE_SECRET: str
    # STRIPE_WEBHOOK_SECRET: str
    # Celery
    # CELERY_BROKER_URL: str
    # CELERY_RESULT_BACKEND: str

    # Auth
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24*7
    ALLOWED_SUPERUSER_EMAILS: List[str] = ['mwschulte23@gmail.com']
    # LLM
    # ANTHROPIC_API_KEY: str
    # FIREWORKS_API_KEY: str
    # HELICONE_API_KEY: str
    # TAVILY_API_KEY: str
    
    # DB
    DATABASE_URL: str
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 0.1
    DB_ECHO: bool = False
    # DB Pooling...not using yet
    POOL_SIZE: int = 2
    MAX_OVERFLOW: int = 3
    POOL_TIMEOUT: int = 10
    POOL_RECYCLE: int = 300

    ENV: str = 'prod' #os.getenv('ENV', 'dev')

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")

    def set_db_url(self):
        if self.ENV == 'prod':
            self.DATABASE_URL = os.getenv('DATABASE_URL')
        else:
            self.DATABASE_URL = os.getenv('DEV_DATABASE_URL')


settings = Settings()
settings.set_db_url()