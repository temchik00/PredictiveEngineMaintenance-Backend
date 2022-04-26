import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import Depends, status, HTTPException, UploadFile
from sqlalchemy.orm import Session
from typing import List, Dict
import tables
from database import get_session
from settings import settings


class PredictionService:
    def __init__(self, session: Session = Depends(get_session)):
        self.session = session
        self.session.autoflush = False
        tf.config.set_visible_devices([], 'GPU')

    def __get_engine_top_cycles_ids(self, engine_id: int) -> List[int]:
        cycle_ids = self.session.query(tables.Cycle.id)\
            .filter_by(engine_id=engine_id).order_by(tables.Cycle.id.desc())\
            .limit(settings.sequence_size).all()
        if cycle_ids is None or len(cycle_ids) != settings.sequence_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough data to predict expected lifetime"
            )
        return [id for id, in cycle_ids]

    def __get_engine_top_cycles(
        self,
        engine_id: int
    ) -> Dict[int, List[float]]:
        cycle_ids = self.__get_engine_top_cycles_ids(engine_id)
        data = self.session.query(tables.PrincipalComponent)\
            .filter_by(engine_id=engine_id)\
            .filter(tables.PrincipalComponent.cycle_id.in_(cycle_ids))\
            .order_by(tables.PrincipalComponent.id.asc()).all()
        grouped_data = {}
        for component in data:
            cycle_id = component.cycle_id
            if grouped_data.get(cycle_id) is None:
                grouped_data[cycle_id] = [component.value]
            else:
                grouped_data[cycle_id].append(component.value)
        return grouped_data

    def predict_lifetime(self, engine_id: int) -> int:
        engine_exists = self.session.query(tables.Engine.id).\
                        filter_by(id=engine_id).first() is not None
        if not engine_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No engine with id {engine_id}")
        grouped_data = self.__get_engine_top_cycles(engine_id)
        data = np.array(list(grouped_data.values()))\
            .reshape(1, settings.sequence_size, -1)
        with tf.device('/cpu:0'):
            model = keras.models.load_model('./app/files/models/current.h5')
            prediction = (np.rint(model.predict(data)))[0][0]
        return prediction
