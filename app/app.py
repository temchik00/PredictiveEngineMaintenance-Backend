from fastapi import FastAPI
from api import router
from fastapi.middleware.cors import CORSMiddleware
from settings import settings

app = FastAPI(root_path=settings.root_path)
app.include_router(router)

if settings.development_mode:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
