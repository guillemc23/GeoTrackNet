

from enum import Enum
from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    computed_field,
)
from pydantic_settings import BaseSettings


class OptimizedBound(str, Enum):
    elbo = 'elbo'
    # TODO: implement fivo
    # fivo = 'fivo' 

class OneHotBins(BaseModel):
    lat : PositiveInt = Field(300)
    lon : PositiveInt = Field(300)
    sog : PositiveInt = Field(30)
    cog : PositiveInt = Field(72)
    
    @computed_field
    @property
    def total(self) -> PositiveInt:
        return self.lat+self.lon+self.sog+self.cog
    
class DatasetConfig(BaseModel):
    training_pickle : FilePath = None
    validation_pickle : FilePath = None
    test_pickle : FilePath = None
    

    mean_pickle : FilePath
    batch_size : PositiveInt = Field(32)
    shuffle : bool = Field(False)
    encoding_bins : OneHotBins = OneHotBins()

class ModelConfig(BaseModel):
    bound : OptimizedBound = OptimizedBound.elbo
    latent_size : PositiveInt = Field(128)

class TrainingConfig(BaseSettings):
    # input
    dataset : DatasetConfig
    model : ModelConfig = ModelConfig()
    # runtime
    random_seed : PositiveInt = Field(11)
    learning_rate : PositiveFloat = Field(0.0003)
    epochs : PositiveInt = Field(20)
    eval_frequency : PositiveInt = 1
    # output
    log_dir : Path = Field(Path('./chkpt'))