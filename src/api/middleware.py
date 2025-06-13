import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles


def setup_middleware(app: FastAPI) -> str:
    # allow frontend requests (production & local dev)
    origins = [
        "https://temperatureforecaster-frontend-production.up.railway.app",
        "http://localhost:5173"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # for serving static files
    static_path: str = os.path.join(os.path.dirname(__file__), '../../static')
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    return static_path
