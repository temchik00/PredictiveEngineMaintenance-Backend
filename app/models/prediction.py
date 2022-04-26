from pydantic import BaseModel


class ExpectedLifetime(BaseModel):
    expectedLifetime: int
