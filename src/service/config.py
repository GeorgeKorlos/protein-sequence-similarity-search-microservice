from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env")

    model_tag: str
    device: str = "cuda"
    index_path: str
    corpus_path: str
    ids_path: str
    max_batch_size: int = 32
    max_payload_size: int = 50000
    max_top_k: int = 50


settings = Settings()  # type: ignore
