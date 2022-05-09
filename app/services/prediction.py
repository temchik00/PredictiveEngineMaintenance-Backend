import numpy as np
from os import listdir
from os.path import exists
import tensorflow as tf
from tensorflow import keras
from keras import layers
from fastapi import Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.orm import Session
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score
import tables
from database import get_session
from settings import settings
from models.prediction import (TrainParameters, TrainResults,
                               ExpectedLifetimeHistory, CycleLifetime)
from utils.prediction import predict_last_cycle


class PredictionService:
    def __init__(self, session: Session = Depends(get_session)):
        self.session = session
        self.session.autoflush = False

    def predict_lifetime(self, engine_id: int) -> int:
        last_cycle_id = self.session.query(func.max(tables.Cycle.id))\
            .filter_by(engine_id=engine_id).scalar()
        prediction = self.session.query(tables.PredictedCycles.count)\
            .filter_by(engine_id=engine_id, cycle_id=last_cycle_id).scalar()
        if prediction is not None:
            return prediction
        else:
            prediction = predict_last_cycle(self.session, engine_id)
            prediction_obj = tables.PredictedCycles(engine_id=engine_id,
                                                    cycle_id=last_cycle_id,
                                                    count=prediction)
            self.session.add(prediction_obj)
            self.session.commit()
            return prediction

    def __get_PCs_from_range(
        self,
        engine_id: int,
        cycle_id_start: int,
        cycle_id_end: int
    ) -> List[Tuple[21 * (float, )]]:
        PCs = self.session\
            .query(func.array_agg(
                aggregate_order_by(tables.PrincipalComponent.value,
                                   tables.PrincipalComponent.id.asc())
            ))\
            .filter_by(engine_id=engine_id)\
            .filter(tables.PrincipalComponent.cycle_id >= cycle_id_start,
                    tables.PrincipalComponent.cycle_id <= cycle_id_end)\
            .order_by(tables.PrincipalComponent.cycle_id.asc())\
            .group_by(tables.PrincipalComponent.cycle_id)\
            .all()
        PCs = [row for row, in PCs]
        return PCs

    def __get_failed_ids(
        self,
        for_testing: bool = False
    ) -> Dict[int, List[Tuple[int, int]]]:
        engine_ids = self.session.query(tables.Engine.id)\
            .filter_by(has_failed=True, for_testing=for_testing).all()
        ids = {}
        for engine_id, in engine_ids:
            failure_points = self.session.query(tables.FailurePoint.cycle_id)\
                .filter_by(engine_id=engine_id)\
                .order_by(tables.FailurePoint.cycle_id.asc()).all()
            failure_points = [point for point, in failure_points]
            cycle_ids = []
            if failure_points[0] >= settings.sequence_size:
                cycle_ids.append((0, failure_points[0]))
            for i in range(1, len(failure_points)):
                failure_point = failure_points[i]
                previous_point = failure_points[i - 1]
                if failure_point - previous_point < settings.sequence_size:
                    continue
                cycle_ids.append((previous_point + 1, failure_point))
            ids[engine_id] = cycle_ids
        return ids

    def __get_remaining_cycles_from_range(
        self,
        engine_id: int,
        id_start: int,
        id_end: int
    ) -> List[int]:
        remaining_cycles = self.session.query(tables.RemainingCycles.count)\
            .filter_by(engine_id=engine_id)\
            .filter(tables.RemainingCycles.cycle_id >= id_start,
                    tables.RemainingCycles.cycle_id <= id_end)\
            .order_by(tables.RemainingCycles.cycle_id.asc()).all()
        return [count for count, in remaining_cycles]

    def __create_dataset(
        self,
        ids: Dict[int, List[Tuple[int, int]]],
        params: TrainParameters,
        for_testing: bool = False
    ) -> tf.data.Dataset:
        sequence_size = settings.sequence_size

        def generator():
            for engine_id, cycle_ids in ids.items():
                for start_id, end_id in cycle_ids:
                    features = self.__get_PCs_from_range(engine_id, start_id,
                                                         end_id)
                    features = np.array(features)
                    labels = self.__get_remaining_cycles_from_range(engine_id,
                                                                    start_id,
                                                                    end_id)
                    labels = np.array(labels)
                    for i in range(features.shape[0] - sequence_size - 1):
                        yield (features[i: i + sequence_size],
                               labels[i + sequence_size])
        n_features = 20
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(settings.sequence_size, n_features),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
        )
        if not for_testing:
            dataset = dataset.batch(params.batchSize)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, n_features

    def __create_neural_network(
        self,
        n_features: int,
        learning_rate: float = 1e-4,
        dropout: float = 0.25
    ):
        model = keras.models.Sequential([
            layers.InputLayer((settings.sequence_size, n_features)),
            layers.LSTM(100,
                        kernel_regularizer=tf.keras.regularizers.L1(0.08),
                        return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(50,
                        kernel_regularizer=tf.keras.regularizers.L1(0.08)),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(12,
                         kernel_regularizer=tf.keras.regularizers.L1(0.08)),
            layers.Dense(1, activation='relu')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def __train_network(
        self,
        dataset,
        n_features: int,
        params: TrainParameters
    ):
        model = self.__create_neural_network(n_features, params.learningRate,
                                             params.dropout)
        model.fit(dataset, verbose=0, epochs=params.epochs)
        return model

    def __test_network(self, model, dataset, labels) -> int:
        predictions = np.rint(model.predict(dataset))
        predictions = predictions.reshape((predictions.shape[0]))
        return r2_score(labels, predictions)

    def __save_model(self, model, as_current: bool = False) -> str:
        filedir = './app/files/models/'
        if as_current:
            filename = "current.h5"
        else:
            files = listdir(filedir)
            n_models = 0
            for file in files:
                if file.startswith('model_') and file.endswith('.h5'):
                    n_models += 1
            filename = f"model_{n_models + 1}.h5"
        name = filedir + filename
        model.save(name)
        return filename

    def train_network(self, params: TrainParameters) -> TrainResults:
        ids = self.__get_failed_ids(for_testing=False)
        train_dataset, n_features = self.__create_dataset(ids, params, False)
        ids = self.__get_failed_ids(for_testing=True)
        test_dataset, n_features = self.__create_dataset(ids, params, True)
        test_labels = np.array([label for _, label in test_dataset])
        test_dataset = test_dataset.batch(params.batchSize)
        model = self.__train_network(train_dataset, n_features, params)
        model_name = self.__save_model(model)
        new_score = self.__test_network(model, test_dataset, test_labels)
        if exists('./app/files/models/current.h5'):
            old_model = keras.models.load_model(
                './app/files/models/current.h5'
            )
            old_score = self.__test_network(old_model, test_dataset,
                                            test_labels)
            if new_score > old_score:
                self.__save_model(model, as_current=True)
                return TrainResults(modelName=model_name, isBetter=True,
                                    score=new_score)
            else:
                return TrainResults(modelName=model_name, isBetter=False,
                                    score=new_score)
        else:
            self.__save_model(model, as_current=True)
            return TrainResults(modelName=model_name, isBetter=True,
                                score=new_score)

    def get_prediction_history(
        self,
        engine_id: int
    ) -> ExpectedLifetimeHistory:
        engine_exists = self.session.query(tables.Engine.id)\
            .filter_by(id=engine_id).first() is not None
        if not engine_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No engine with id {engine_id}")
        predictions = self.session.query(tables.PredictedCycles)\
            .filter_by(engine_id=engine_id)\
            .order_by(tables.PredictedCycles.cycle_id.asc()).all()
        history = []
        for prediction in predictions:
            history.append(CycleLifetime(cycleId=prediction.cycle_id,
                                         lifetime=prediction.count))
        if len(history) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prediction history for engine with id {engine_id}"
            )
        return ExpectedLifetimeHistory(history=history)
