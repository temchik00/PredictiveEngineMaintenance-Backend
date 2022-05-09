from typing import List
from pydantic import BaseModel


class ExpectedLifetime(BaseModel):
    expectedLifetime: int


class TrainParameters(BaseModel):
    epochs: int = 10
    batchSize: int = 256
    learningRate: float = 1e-4
    dropout: float = 0.25


class TrainResults(BaseModel):
    modelName: str
    isBetter: bool
    score: float


class CycleLifetime(BaseModel):
    lifetime: int
    cycleId: int


class ExpectedLifetimeHistory(BaseModel):
    history: List[CycleLifetime]
