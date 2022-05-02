import numpy as np
from os import listdir
from os.path import exists
import tensorflow as tf
from tensorflow import keras
from keras import layers
from fastapi import Depends, status, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict
from sklearn.metrics import r2_score
import tables
from database import get_session
from settings import settings
from models.prediction import TrainParameters, TrainResults


class PredictionService:
    def __init__(self, session: Session = Depends(get_session)):
        self.session = session
        self.session.autoflush = False

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
        if not exists('./app/files/models/current.h5'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No trained model to predict lifetime"
            )
        engine_exists = self.session.query(tables.Engine.id)\
            .filter_by(id=engine_id).first() is not None
        if not engine_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No engine with id {engine_id}")
        grouped_data = self.__get_engine_top_cycles(engine_id)
        data = np.array(list(grouped_data.values()))\
            .reshape(1, settings.sequence_size, -1)
        with tf.device('/GPU:0'):
            model = keras.models.load_model('./app/files/models/current.h5')
            prediction = (np.rint(model.predict(data)))[0][0]
        return prediction

    def __get_PCs(
        self,
        engine_id: int,
        cycle_ids: List[int]
    ) -> List[List[float]]:
        PCs = self.session.query(tables.PrincipalComponent)\
            .filter_by(engine_id=engine_id)\
            .filter(tables.PrincipalComponent.cycle_id.in_(cycle_ids))\
            .order_by(tables.PrincipalComponent.cycle_id.asc(),
                      tables.PrincipalComponent.id.asc()).all()
        splitted_pcs = {}
        for PC in PCs:
            if splitted_pcs.get(PC.cycle_id) is None:
                splitted_pcs[PC.cycle_id] = [PC.value]
            else:
                splitted_pcs[PC.cycle_id].append(PC.value)
        return list(splitted_pcs.values())

    def __get_remaining_cycles(
        self,
        engine_id: int,
        cycle_ids: List[int]
    ) -> List[float]:
        remaining_cycles = self.session.query(tables.RemainingCycles.count)\
            .filter_by(engine_id=engine_id)\
            .filter(tables.RemainingCycles.cycle_id.in_(cycle_ids))\
            .order_by(tables.RemainingCycles.engine_id.asc(),
                      tables.RemainingCycles.cycle_id.asc()).all()
        return [count for count, in remaining_cycles]

    def __get_failed_data(self, for_testing: bool = False):
        engine_ids = self.session.query(tables.Engine.id)\
            .filter_by(has_failed=True, for_testing=for_testing).all()
        features = []
        labels = []
        for engine_id, in engine_ids:
            cycle_ids = self.session.query(tables.Cycle.id)\
                .filter_by(engine_id=engine_id).all()
            if len(cycle_ids) < settings.sequence_size:
                continue
            cycle_ids = [id for id, in cycle_ids]
            PCs = self.__get_PCs(engine_id, cycle_ids)
            features += PCs
            remaining_cycles = self.__get_remaining_cycles(engine_id,
                                                           cycle_ids)
            labels += remaining_cycles
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def __get_training_data(self):
        features, labels = self.__get_failed_data(for_testing=False)
        if labels.shape[0] < 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough data to train model"
            )
        return features, labels

    def __get_testing_data(self):
        features, labels = self.__get_failed_data(for_testing=True)
        if labels.shape[0] < 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough data to test model"
            )
        return features, labels

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

    def __create_train_dataset(self, params):
        features, labels = self.__get_training_data()
        n_features = features.shape[1]

        def generate_sequences(features, labels):
            sequence_size = settings.sequence_size
            for timestep in range(features.shape[0] - sequence_size - 1):
                yield features[timestep: timestep + sequence_size],\
                    labels[timestep + sequence_size - 1]

        dataset = tf.data.Dataset.from_generator(
            generate_sequences,
            output_signature=(
                tf.TensorSpec(shape=(settings.sequence_size, n_features),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
            args=(features, labels)
        )
        dataset = dataset.batch(params.batchSize)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, n_features

    def __create_test_dataset(self, features, labels):
        n_features = features.shape[1]

        def generate_sequences(features):
            sequence_size = settings.sequence_size
            for timestep in range(features.shape[0] - sequence_size - 1):
                yield features[timestep: timestep + sequence_size],\
                    labels[timestep + sequence_size - 1]

        dataset = tf.data.Dataset.from_generator(
            generate_sequences,
            output_signature=(
                tf.TensorSpec(shape=(settings.sequence_size, n_features),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
            args=(features, )
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

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
        train_dataset, n_features = self.__create_train_dataset(params)
        test_features, test_labels = self.__get_testing_data()
        test_dataset = self.__create_test_dataset(test_features, test_labels)
        test_labels = np.array([y for _, y in test_dataset])
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
