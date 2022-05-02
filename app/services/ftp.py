from fastapi import HTTPException, status
from os import getcwd
from os.path import join, isfile


class FtpService:
    def __init__(self):
        self.filedir = join(getcwd(), 'app', 'files', 'models')

    def get_path(self, filename: str):
        filepath: str = join(self.filedir, filename)
        if isfile(filepath):
            return filepath
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
