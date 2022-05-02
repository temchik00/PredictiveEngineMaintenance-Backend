from fastapi import APIRouter
from .data import router as data_router
from .prediction import router as prediction_router
from .ftp import router as ftp_router

router = APIRouter()
router.include_router(data_router)
router.include_router(prediction_router)
router.include_router(ftp_router)
