import numpy as np
from os.path import exists
import tensorflow as tf
from tensorflow import keras
from fastapi import status, HTTPException
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.orm import Session
from typing import List, Dict, Tuple
import tables
from settings import settings


def get_PCs(
    session: Session,
    engine_id: int,
    cycle_ids: List[int]
) -> List[Tuple[21 * (float, )]]:
    PCs = session\
        .query(func.array_agg(
            aggregate_order_by(tables.PrincipalComponent.value,
                               tables.PrincipalComponent.id.asc())
        ))\
        .filter_by(engine_id=engine_id)\
        .filter(tables.PrincipalComponent.cycle_id.in_(cycle_ids))\
        .order_by(tables.PrincipalComponent.cycle_id.asc())\
        .group_by(tables.PrincipalComponent.cycle_id)\
        .all()
    PCs = [row for row, in PCs]
    return PCs


def get_engine_top_cycles_ids(session: Session, engine_id: int) -> List[int]:
    failure_points = session.query(tables.FailurePoint.cycle_id)\
        .filter_by(engine_id=engine_id)\
        .order_by(tables.FailurePoint.cycle_id.desc()).all()
    if failure_points and len(failure_points) > 0:
        last_failure_point = failure_points[-1][0]
    else:
        last_failure_point = 0
    last_cycle_id = session.query(func.max(tables.Cycle.id))\
        .filter_by(engine_id=engine_id).scalar()
    if last_failure_point == last_cycle_id:
        if len(failure_points) == 1:
            last_failure_point = 0
        else:
            last_failure_point = failure_points[-2][0]
    cycle_ids = session.query(tables.Cycle.id)\
        .filter_by(engine_id=engine_id)\
        .filter(tables.Cycle.id > last_failure_point)\
        .order_by(tables.Cycle.id.desc())\
        .limit(settings.sequence_size).all()
    if cycle_ids is None or len(cycle_ids) != settings.sequence_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough data to predict expected lifetime"
        )
    return [id for id, in cycle_ids]


def get_engine_top_cycles(
    session: Session,
    engine_id: int
) -> Dict[int, List[float]]:
    cycle_ids = get_engine_top_cycles_ids(session, engine_id)
    PCs = get_PCs(session, engine_id, cycle_ids)
    return PCs


def predict_last_cycle(session: Session, engine_id: int) -> int:
    if not exists('./app/files/models/current.h5'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No trained model to predict lifetime"
        )
    engine_exists = session.query(tables.Engine.id)\
        .filter_by(id=engine_id).first() is not None
    if not engine_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No engine with id {engine_id}")
    data = np.array(get_engine_top_cycles(session, engine_id))\
        .reshape(1, settings.sequence_size, -1)

    model = keras.models.load_model('./app/files/models/current.h5')
    prediction = int(np.rint(model.predict(data)).astype(int)[0][0])
    return prediction
