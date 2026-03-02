from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_url: str
    database_name: str = "She_Health"
    
    class Config:
        env_file = ".env"

settings = Settings()
