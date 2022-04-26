from pydantic import BaseModel
from typing import List


class CycleBase(BaseModel):
    setting1: float
    setting2: float
    setting3: float
    sensorValues: List[float]


class CycleAdd(CycleBase):
    pass


class Engine(BaseModel):
    id: int
    hasFailed: bool
