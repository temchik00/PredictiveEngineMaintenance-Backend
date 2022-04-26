from fastapi import APIRouter, Depends, File, UploadFile, status
from services.prediction import PredictionService


router = APIRouter(prefix='/prediction')


@router.get('/{engine_id}', status_code=status.HTTP_200_OK)
def predict_lifetime(
    engine_id: int,
    service: PredictionService = Depends()
):
    ...


@router.put('/model', status_code=status.HTTP_200_OK)
def update_model(
    model: UploadFile = File(...),
    service: PredictionService = Depends(),
):
    ...


@router.post('/train', status_code=status.HTTP_201_CREATED)
def retrain_model(
    service: PredictionService = Depends(),
):
    ...
