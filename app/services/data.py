import io
import numpy as np
import pandas as pd
from fastapi import Depends, status, HTTPException, UploadFile
from sqlalchemy.orm import Session
from typing import List, Dict
from database import get_session
import tables
from pickle import load
from models.data import CycleAdd
from settings import settings


class DataService:
    def __init__(self, session: Session = Depends(get_session)):
        self.session = session
        self.session.autoflush = False
        with open('./app/files/other/pca.pkl', 'rb') as file:
            self.PCA = load(file)

    def __group_sensors(
        self,
        sensors: List[tables.Sensor]
    ) -> Dict[int, List[float]]:
        splitted_sensors = {}
        for sensor in sensors:
            if splitted_sensors.get(sensor.cycle_id) is None:
                splitted_sensors[sensor.cycle_id] = [sensor.value]
            else:
                splitted_sensors[sensor.cycle_id].append(sensor.value)
        return splitted_sensors

    def __calc_sensors_avg(
        self,
        sensors: Dict[int, List[float]]
    ) -> List[float]:
        sensors = np.array(list(sensors.values()))
        return np.average(sensors, axis=0).tolist()

    def __calc_sensors_std(
        self,
        sensors: Dict[int, List[float]]
    ) -> List[float]:
        sensors = np.array(list(sensors.values()))
        return np.std(sensors, axis=0).tolist()

    def __add_principal_components(
        self,
        engine_id: int,
        cycle_id: int,
        pcs: List[float]
    ) -> List[tables.PrincipalComponent]:
        principal_components = []
        for pc in pcs:

            principal_component =\
                tables.PrincipalComponent(engine_id=engine_id,
                                          cycle_id=cycle_id,
                                          value=pc)
            principal_components.append(principal_component)
        return principal_components

    def __process_last_cycle(
        self,
        engine_id: int
    ) -> List[tables.PrincipalComponent]:
        last_failure_point = self.session.query(tables.FailurePoint.cycle_id).\
            filter_by(engine_id=engine_id).\
            order_by(tables.FailurePoint.cycle_id.desc()).first()
        if last_failure_point is None:
            cycles_ids = self.session.query(tables.Cycle.id).\
                filter_by(engine_id=engine_id).\
                order_by(tables.Cycle.id.desc()).\
                limit(settings.window_size).all()
        else:
            cycles_ids = self.session.query(tables.Cycle.id).\
                filter_by(engine_id=engine_id).\
                order_by(tables.Cycle.id.desc()).\
                filter(tables.Cycle.id > last_failure_point).\
                limit(settings.window_size).all()
        cycles_ids = [cycle_id for cycle_id, in cycles_ids]
        sensors = self.session.query(tables.Sensor).\
            filter_by(engine_id=engine_id).\
            order_by(tables.Sensor.cycle_id.desc(), tables.Sensor.id.asc()).\
            filter(tables.Sensor.cycle_id.in_(cycles_ids)).all()
        splitted_sensors = self.__group_sensors(sensors)
        sensors_avg = self.__calc_sensors_avg(splitted_sensors)
        sensors_std = self.__calc_sensors_std(splitted_sensors)
        last_cycle = self.session.query(tables.Cycle).\
            filter_by(engine_id=engine_id).order_by(tables.Cycle.id.desc()).\
            first()
        cycle_data = [last_cycle.id, last_cycle.setting1,
                      last_cycle.setting2, last_cycle.setting3]
        last_sensor_values = self.session.query(tables.Sensor.value).\
            filter_by(engine_id=engine_id, cycle_id=last_cycle.id).\
            order_by(tables.Sensor.id.asc()).all()
        last_sensor_values = [value for value, in last_sensor_values]
        cycle_data += last_sensor_values
        cycle_data += sensors_avg
        cycle_data += sensors_std
        cycle_data = np.array(cycle_data).reshape(1, -1)
        principal_components = self.PCA.transform(cycle_data)\
            .reshape(-1).tolist()
        return self.__add_principal_components(engine_id, last_cycle.id,
                                               principal_components)

    def __prepare_sensor_values(
        self,
        engine_id: int,
        cycle_id: int,
        sensor_values: List[float]
    ) -> List[tables.Sensor]:
        sensors = []
        for sensor_value in sensor_values:
            sensor = tables.Sensor(cycle_id=cycle_id,
                                   engine_id=engine_id,
                                   value=sensor_value)
            sensors.append(sensor)
        return sensors

    def __add_failure_point(
        self,
        engine_id: int,
        cycle_id: int
    ) -> tables.FailurePoint:
        return tables.FailurePoint(cycle_id=cycle_id,
                                   engine_id=engine_id)

    def __create_remaining_cycles(
        self,
        engine_id: int
    ) -> List[tables.RemainingCycles]:
        last_failure_point = self.session.query(tables.FailurePoint.cycle_id).\
            filter_by(engine_id=engine_id).\
            order_by(tables.FailurePoint.cycle_id.desc()).first()
        remaining_cycles = []
        last_cycle = self.session.query(tables.Cycle.id).\
            filter_by(engine_id=engine_id).order_by(tables.Cycle.id.desc()).\
            first()
        self.__process_last_cycle(engine_id)
        if last_failure_point is None:
            cycle_ids = self.session.query(tables.Cycle.id).\
                filter_by(engine_id=engine_id).all()
        else:
            cycle_ids = self.session.query(tables.Cycle.id).\
                filter_by(engine_id=engine_id).\
                filter(tables.Cycle.id > last_failure_point).all()
        for cycle_id in cycle_ids:
            remaining_cycles.append(
                tables.RemainingCycles(
                    engine_id=engine_id,
                    cycle_id=cycle_id[0],
                    count=last_cycle[0]-cycle_id[0]
                )
            )
        return remaining_cycles

    def __create_cycle(
        self,
        engine_id: int,
        cycle_add: CycleAdd,
    ) -> tables.Cycle:
        id = self.session.query(tables.Cycle.id).\
             filter_by(engine_id=engine_id).order_by(tables.Cycle.id.desc()).\
             first()
        if id is None:
            id = 1
        else:
            id = id[0] + 1
        return tables.Cycle(id=id,
                            engine_id=engine_id,
                            setting1=cycle_add.setting1,
                            setting2=cycle_add.setting2,
                            setting3=cycle_add.setting3)

    def add_cycle(
        self,
        engine_id: int,
        cycle_add: CycleAdd,
        has_failed: bool
    ) -> None:
        engine_exists = self.session.query(tables.Engine.id).\
                        filter_by(id=engine_id).first() is not None
        if not engine_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No engine with id {engine_id}")
        cycle = self.__create_cycle(engine_id, cycle_add)
        self.session.add(cycle)
        self.session.flush()
        self.session.refresh(cycle)
        sensors = self.__prepare_sensor_values(engine_id,
                                               cycle.id,
                                               cycle_add.sensorValues)
        self.session.add_all(sensors)
        self.session.flush()
        pcs = self.__process_last_cycle(engine_id)
        self.session.add_all(pcs)
        if has_failed:
            remaining_cycles = self.__create_remaining_cycles(engine_id)
            self.session.add_all(remaining_cycles)
            failure_point = self.__add_failure_point(engine_id, cycle.id)
            self.session.add(failure_point)
        self.session.commit()

    def __create_engine(self, has_failed: bool = False) -> tables.Engine:
        engine = tables.Engine(has_failed=has_failed)
        return engine

    def add_engine(self) -> tables.Engine:
        engine = self.__create_engine()
        self.session.add(engine)
        self.session.commit()
        self.session.refresh(engine)
        return engine

    async def __get_data_from_file(self, file: UploadFile) -> pd.DataFrame:
        sensor_names = [f'sensor{i}' for i in range(1, 22)]
        data_names = ['engine_id', 'cycle', 'setting1',
                      'setting2', 'setting3', *sensor_names]
        content = await file.read()
        content = str(content, 'utf-8')
        content = io.StringIO(content)
        data = pd.read_csv(content, names=data_names,
                           sep=' ', index_col=False)
        sensor_avg = data.groupby('engine_id')[sensor_names]\
            .rolling(min_periods=1, window=settings.window_size).mean()\
            .set_axis([f'avg{i}' for i in range(1, 22)],
                      axis=1, inplace=False)\
            .reset_index().drop(columns=['engine_id', 'level_1'])
        sensor_std = data.groupby('engine_id')[sensor_names]\
            .rolling(min_periods=1, window=settings.window_size).std(ddof=0)\
            .set_axis([f'std{i}' for i in range(1, 22)],
                      axis=1, inplace=False)\
            .reset_index().drop(columns=['engine_id', 'level_1'])
        cycle_data = data[['engine_id', 'cycle']]
        last_cycles = data.groupby(['engine_id'])['cycle'].max()\
            .rename('last_cycle')
        tmp = pd.merge(cycle_data, last_cycles, on='engine_id')
        time_to_fail = (tmp['last_cycle'] - tmp['cycle'])\
            .rename('cycles_to_fail')
        full_features = data.join(sensor_avg)
        full_features = full_features.join(sensor_std)
        full_features = full_features.join(time_to_fail)
        return full_features

    async def load_from_file(self, file: UploadFile):
        sensor_names = [f'sensor{i}' for i in range(1, 22)]
        data = await self.__get_data_from_file(file)
        for _, engine_data in data.groupby('engine_id'):
            engine = self.__create_engine(True)

            self.session.add(engine)
            self.session.flush()
            self.session.refresh(engine)
            engine_id = engine.id
            cycles = []
            remaining_cycles = []
            sensors = []
            for _, cycle_data in engine_data.iterrows():
                cycle_id = cycle_data['cycle']
                cycle = tables.Cycle(
                    id=cycle_id,
                    engine_id=engine_id,
                    setting1=cycle_data['setting1'],
                    setting2=cycle_data['setting2'],
                    setting3=cycle_data['setting3']
                )
                cycles.append(cycle)
                remaining_cycle =\
                    tables.RemainingCycles(engine_id=engine_id,
                                           cycle_id=cycle_id,
                                           count=cycle_data['cycles_to_fail'])
                remaining_cycles.append(remaining_cycle)
                for sensor_name in sensor_names:
                    sensor_data = cycle_data[sensor_name]
                    sensor = tables.Sensor(cycle_id=cycle_id,
                                           engine_id=engine_id,
                                           value=sensor_data)
                    sensors.append(sensor)
            self.session.add_all(cycles)
            self.session.add_all(remaining_cycles)
            self.session.add_all(sensors)
            last_cycle = cycle_data['cycle'].max()
            failure_point = tables.FailurePoint(engine_id=engine_id,
                                                cycle_id=last_cycle)
            self.session.add(failure_point)
            data_for_pca = engine_data.drop(columns=['engine_id',
                                                     'cycles_to_fail'])
            pcs_data = self.PCA.transform(data_for_pca)

            pcs = []
            for i in range(pcs_data.shape[0]):
                cycle_id = i + 1
                for component in pcs_data[i]:
                    pc = tables.PrincipalComponent(cycle_id=cycle_id,
                                                   engine_id=engine_id,
                                                   value=component)
                    pcs.append(pc)
            self.session.add_all(pcs)
        self.session.commit()
        return
