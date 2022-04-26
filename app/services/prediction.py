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
