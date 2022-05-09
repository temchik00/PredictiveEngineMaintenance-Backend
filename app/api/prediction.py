from fastapi import APIRouter, Depends, status
from services.prediction import PredictionService
from models.prediction import (ExpectedLifetime, TrainParameters,
                               TrainResults, ExpectedLifetimeHistory)


router = APIRouter(prefix='/prediction')


@router.get(
    '/{engine_id}/',
    status_code=status.HTTP_200_OK,
    response_model=ExpectedLifetime
)
def predict_lifetime(
    engine_id: int,
    service: PredictionService = Depends()
):
    prediction = service.predict_lifetime(engine_id)
    return ExpectedLifetime(expectedLifetime=prediction)


@router.post(
    '/train/',
    status_code=status.HTTP_201_CREATED,
    response_model=TrainResults
)
def retrain_model(
    parameters: TrainParameters,
    service: PredictionService = Depends(),
):
    return service.train_network(parameters)


@router.get('/history/{engine_id}/', status_code=status.HTTP_200_OK,
            response_model=ExpectedLifetimeHistory)
def get_prediction_history(
    engine_id: int,
    service: PredictionService = Depends()
):
    return service.get_prediction_history(engine_id)
