from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import health, events

app = FastAPI(
    title="Epsilon Backend",
    version="0.1.0",
    description="Backend for Concept Epsilon (transient leak localization).",
)

# CORS:
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(events.router, prefix="/api")
