from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    mongodb_url: str
    database_name: str = "She_Health"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
