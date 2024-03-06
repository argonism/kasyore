import os
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .encoder import Retriever

is_dev = (
    os.environ.get("FOTLA_ENV", "dev") == "dev"
    or os.environ.get("FOTLA_ENV", "dev") == "development"
)


def start_api(retriever: Retriever, host: str = "0.0.0.0", port: int = 8000) -> None:
    app = load_fastapi_app(retriever)

    uvicorn.run(app, host=host, port=port)


def load_fastapi_app(retriever: Retriever) -> FastAPI:
    app = FastAPI()
    setup_api_endpoint(app, retriever)
    return app


def setup_api_endpoint(app: FastAPI, retriever: Retriever) -> None:
    class SearchRequest(BaseModel):
        query: str
        topk: int = 200
        from_: int = 0
        size: int = 10
        hybrid: bool = True
        search_fields: List[str] = ["subject_number", "subject_number", "overview"]

    @app.post("/search")
    async def search(request: SearchRequest) -> Dict[str, Any]:
        result = retriever.retrieve(
            [request.query],
            top_k=request.topk,
            from_=request.from_,
            size=request.size,
            hybrid=request.hybrid,
            search_fields=request.search_fields,
        )
        return {"status": "success", "result": result}
