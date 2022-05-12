import uvicorn
from settings import settings

if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        host='0.0.0.0',
        port=settings.server_port,
        reload=settings.development_mode
    )
