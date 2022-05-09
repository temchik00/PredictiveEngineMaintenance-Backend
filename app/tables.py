from sqlalchemy import (
    Column,
    Integer,
    Boolean,
    Float,
    ForeignKey,
    ForeignKeyConstraint
)
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Engine(Base):
    __tablename__ = 'engine'
    id = Column(Integer, primary_key=True)
    has_failed = Column(Boolean, nullable=False, default=False)
    for_testing = Column(Boolean, nullable=False, default=False)


class Cycle(Base):
    __tablename__ = 'cycle'
    id = Column(Integer, primary_key=True)
    engine_id = Column(Integer, ForeignKey('engine.id'), primary_key=True)
    setting1 = Column(Float, nullable=False)
    setting2 = Column(Float, nullable=False)
    setting3 = Column(Float, nullable=False)


class PrincipalComponent(Base):
    __tablename__ = 'principal_component'
    id = Column(Integer, primary_key=True)
    cycle_id = Column(Integer, nullable=False)
    engine_id = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    ForeignKeyConstraint(
        ['cycle_id', 'engine_id'],
        ['cycle.id', 'cycle.engine_id']
    )


class Sensor(Base):
    __tablename__ = 'sensor'
    id = Column(Integer, primary_key=True)
    cycle_id = Column(Integer, nullable=False)
    engine_id = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    ForeignKeyConstraint(
        ['cycle_id', 'engine_id'],
        ['cycle.id', 'cycle.engine_id']
    )


class FailurePoint(Base):
    __tablename__ = 'failure_point'
    cycle_id = Column(Integer, primary_key=True)
    engine_id = Column(
        Integer,
        primary_key=True
    )
    ForeignKeyConstraint(
        ['cycle_id', 'engine_id'],
        ['cycle.id', 'cycle.engine_id']
    )
    ForeignKeyConstraint(
        ['engine_id'],
        ['engine.id']
    )


class RemainingCycles(Base):
    __tablename__ = 'remaining_cycles'
    engine_id = Column(Integer, primary_key=True)
    cycle_id = Column(Integer, primary_key=True)
    count = Column(Integer, nullable=False)
    ForeignKeyConstraint(
        ['cycle_id', 'engine_id'],
        ['cycle.id', 'cycle.engine_id']
    )


class PredictedCycles(Base):
    __tablename__ = 'predicted_cycles'
    engine_id = Column(Integer, primary_key=True)
    cycle_id = Column(Integer, primary_key=True)
    count = Column(Integer, nullable=False)
    ForeignKeyConstraint(
        ['cycle_id', 'engine_id'],
        ['cycle.id', 'cycle.engine_id']
    )
