from typing import Optional
from fastapi import APIRouter, Depends, status, Response, UploadFile, File
from models.data import CycleAdd, Engine
from services.data import DataService


router = APIRouter(prefix='/data')


@router.post('/loadFromCSV/', status_code=status.HTTP_204_NO_CONTENT)
async def load_from_CSV(
    file: UploadFile = File(...),
    for_testing: Optional[bool] = False,
    service: DataService = Depends(),
):
    await service.load_from_file(file, for_testing)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post('/cycle/{engine_id}/', status_code=status.HTTP_204_NO_CONTENT)
def add_cycle(
    engine_id: int,
    has_failed: bool,
    cycle: CycleAdd,
    service: DataService = Depends(),
):
    service.add_cycle(engine_id, cycle, has_failed)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post('/engine/',
             status_code=status.HTTP_201_CREATED,
             response_model=Engine)
def add_engine(service: DataService = Depends()):
    engine = service.add_engine()
    return Engine(id=engine.id, hasFailed=engine.has_failed)


@router.get('/engine/{engine_id}/',
            status_code=status.HTTP_200_OK,
            response_model=Engine)
def get_engine(engine_id: int, service: DataService = Depends()):
    engine = service.get_engine(engine_id)
    return Engine(id=engine.id, hasFailed=engine.has_failed)
