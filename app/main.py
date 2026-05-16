"""FastAPI application factory."""
from fastapi import FastAPI

from app.config import ADAPTER_NAME, MODEL_NAME
from app.routes.ollama import router as ollama_router
from app.routes.openai import router as openai_router


def create_app() -> FastAPI:
    application = FastAPI(
        title=f"Continue Adapter ({ADAPTER_NAME} / {MODEL_NAME})",
    )

    # Register route groups
    application.include_router(openai_router)
    application.include_router(ollama_router)

    @application.get("/")
    async def root():
        return {
            "status": "ok",
            "adapter": ADAPTER_NAME,
            "model": MODEL_NAME,
            "message": "Continue adapter — OpenAI & Ollama compatible (one model step per request; tools run in IDE).",
        }

    return application


app = create_app()
