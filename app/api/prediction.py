from fastapi import APIRouter, Depends, File, UploadFile, status
from services.prediction import PredictionService
from models.prediction import ExpectedLifetime


router = APIRouter(prefix='/prediction')


@router.get(
    '/{engine_id}',
    status_code=status.HTTP_200_OK,
    response_model=ExpectedLifetime
)
def predict_lifetime(
    engine_id: int,
    service: PredictionService = Depends()
):
    prediction = service.predict_lifetime(engine_id)
    return ExpectedLifetime(expectedLifetime=prediction)


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
