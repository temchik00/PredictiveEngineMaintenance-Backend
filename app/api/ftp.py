from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from services.ftp import FtpService


router = APIRouter(
    prefix='/file'
)


@router.get("/{filename}/")
def get_file(filename: str, service: FtpService = Depends()):
    return FileResponse(path=service.get_path(filename))
