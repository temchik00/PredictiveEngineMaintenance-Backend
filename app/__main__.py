import uvicorn
from settings import settings

if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        port=settings.server_port,
        reload=True
    )
