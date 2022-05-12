from pydantic import BaseSettings


class Settings(BaseSettings):
    root_path: str = ''
    development_mode: bool = True
    server_port: int
    database_url: str
    window_size: int = 5
    sequence_size: int


settings = Settings(
    _env_file='.env',
    _env_file_encoding='utf-8'
)
