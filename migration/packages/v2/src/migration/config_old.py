from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class TrainingSettings(BaseSettings):
    bound : str = Field('elbo')
    latent_size : int = Field(64)
    log_dir : Path = Field(Path('./chkpt'))
    batch_size : int = Field(32)
    random_seed : int = Field(11)
    learning_rate : float = Field(0.0003)

