import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router
from src.data_loader import data_loader

app = FastAPI(
    title="Graph RAG vs PageRank Evaluation Platform",
    description="Bachelor Thesis Research Platform: Evaluating Lexical PageRank, Dense RAG, and Fusion Models.",
    version="2.0.0"
)

# Enable CORS for Codespaces cloud port forwarding and cross-origin calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST API endpoints under /api namespace
app.include_router(api_router, prefix="/api")

# Ensure static directories exist before mounting
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Lightweight health check probe for cloud infrastructure and Azure monitoring.
    Verifies that the server process is responsive and in-memory datasets are loaded.
    """
    return {
        "status": "healthy",
        "service": "Graph RAG vs PageRank Evaluation Platform",
        "datasets_available": list(data_loader.datasets.keys())
    }

# Mount static asset directories
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index_page():
    """Serves the primary Single Page Application interface."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    # Local debugging entrypoint
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)