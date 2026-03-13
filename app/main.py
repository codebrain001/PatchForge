import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.core.storage import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("patchforge").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("patchforge.app")


async def _llm_health_check() -> None:
    """Validate LLM provider on startup so misconfiguration is caught early."""
    import asyncio
    from app.core.llm import is_llm_available, get_active_provider, call_llm

    provider = get_active_provider()
    if not is_llm_available():
        logger.error(
            "CRITICAL: No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env. "
            "The LLM is the core decision engine — the pipeline cannot make calibration or "
            "thickness decisions without it."
        )
        return

    logger.info("LLM provider configured: %s — sending health-check...", provider)
    try:
        _, used = await asyncio.to_thread(
            call_llm,
            "You are a health-check bot.",
            "Reply with exactly: OK",
        )
        logger.info("LLM health-check passed (provider: %s)", used)
    except Exception as e:
        msg = str(e)
        if "billing_not_active" in msg or "billing" in msg.lower():
            logger.error(
                "LLM health-check FAILED: OpenAI billing is not active. "
                "Enable billing at https://platform.openai.com/account/billing"
            )
        elif "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            logger.error(
                "LLM health-check FAILED: %s quota exhausted. "
                "Switch LLM_PROVIDER in .env or wait for quota reset.",
                provider,
            )
        else:
            logger.error("LLM health-check FAILED: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    logger.info("PatchForge v%s starting up", app.version)
    await _llm_health_check()
    yield


app = FastAPI(
    title="PatchForge",
    description="From Photo to Patch: AI-powered 3D-printable repair part generation",
    version="1.0.0",
    lifespan=lifespan,
)

from app.core.exceptions import PatchForgeError


@app.exception_handler(PatchForgeError)
async def patchforge_error_handler(request: Request, exc: PatchForgeError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


from app.api.upload import router as upload_router
from app.api.jobs import router as jobs_router
from app.api.mesh import router as mesh_router
from app.api.frames import router as frames_router
from app.api.printer import router as printer_router
from app.api.prompt import router as prompt_router

app.include_router(upload_router, prefix="/api/v1", tags=["upload"])
app.include_router(prompt_router, prefix="/api/v1", tags=["prompt"])
app.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])
app.include_router(mesh_router, prefix="/api/v1", tags=["mesh"])
app.include_router(frames_router, prefix="/api/v1", tags=["frames"])
app.include_router(printer_router, prefix="/api/v1", tags=["printer"])

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
async def health():
    from app.core.llm import is_llm_available, get_active_provider
    return {
        "status": "ok",
        "version": app.version,
        "llm_available": is_llm_available(),
        "llm_provider": get_active_provider(),
    }


@app.get("/")
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "PatchForge API is running. Visit /docs for Swagger UI."}
