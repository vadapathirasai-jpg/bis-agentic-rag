"""FastAPI application for BIS Sahayak.

Provides REST API endpoints bridging React frontend to the BIS Sahayak Consumer Agent:
- GET /health: Basic health probe
- GET /api/status: Dependency status (API, Qdrant, Gemini)
- POST /api/chat: Consumer Agent query pipeline returning grounded answers and citations
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from app.agents.consumer_agent import ConsumerAgent

# Configure logging
logger = logging.getLogger("bis_sahayak.api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# FastAPI App Instance
app = FastAPI(
    title="BIS Sahayak API",
    description="Backend API service connecting consumer inquiries to the BIS Sahayak Agentic RAG pipeline.",
    version="1.0.0",
)

# CORS Configuration for React development servers
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language question or query for the BIS Consumer Agent",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace only.")
        return cleaned


class HealthResponse(BaseModel):
    status: str
    service: str


class StatusResponse(BaseModel):
    api: bool
    qdrant: bool
    gemini_configured: bool


class ErrorResponse(BaseModel):
    error: bool
    message: str


# Global Agent Singleton
_agent: Optional[ConsumerAgent] = None


def get_agent() -> ConsumerAgent:
    """Retrieve or initialize the ConsumerAgent singleton instance."""
    global _agent
    if _agent is None:
        logger.info("Initializing ConsumerAgent singleton...")
        _agent = ConsumerAgent()
        try:
            _agent.retriever.connect()
            logger.info("ConsumerAgent connected to Qdrant collection '%s'.", _agent.retriever.collection_name)
        except Exception as err:
            logger.warning("ConsumerAgent Qdrant connection deferred or warning: %s", err)
    return _agent


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": True,
            "message": "Invalid request parameters. Please provide a valid, non-empty question message.",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": True,
            "message": str(exc),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "Unable to process the request right now.",
        },
    )


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> Dict[str, str]:
    """Basic health probe endpoint."""
    return {
        "status": "ok",
        "service": "BIS Sahayak",
    }


@app.get("/api/status", response_model=StatusResponse, tags=["Health"])
def dependency_status() -> Dict[str, bool]:
    """Check API and dependency connectivity (Qdrant, Gemini)."""
    agent = get_agent()

    qdrant_ok = False
    try:
        qdrant_ok = agent.retriever.qdrant_store.is_healthy()
    except Exception:
        qdrant_ok = False

    gemini_ok = agent.gemini.is_configured

    return {
        "api": True,
        "qdrant": qdrant_ok,
        "gemini_configured": gemini_ok,
    }


@app.post("/api/chat", tags=["Chat"])
def chat(payload: ChatRequest) -> Dict[str, Any]:
    """Process a user query through the Consumer Agent and return grounded answer with citations."""
    try:
        agent = get_agent()
        result = agent.answer_question(payload.message)
        return result
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": True, "message": str(err)},
        )
    except Exception as err:
        logger.error("Error running ConsumerAgent pipeline: %s", err, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "message": "Unable to process the request right now.",
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
