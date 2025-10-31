"""
Module : app_factory.py
-----------------------
Crée et configure l'application FastAPI pour le Web Monitor :
- CORS pour accès LAN
- sécurité via X-API-Token
- montage statiques (/assets)
- enregistrement routes REST et WebSocket
- initialisation propre du LogCollector (réel ou simulé)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path
from contextlib import asynccontextmanager
import logging
import tempfile
import asyncio

# Collector réel : LogCollector lit les fichiers pipeline.log et kpi.log
from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector as Collector

# Import des routes
from sandbox.web_monitor.common.app.api_routes import register_api_routes
from sandbox.web_monitor.common.app.ws_routes import register_ws_routes


# ------------------------------------------------------------------
# 🧩 Gestion du cycle de vie (startup / shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie complet de l'application FastAPI."""
    log = logging.getLogger("igt.dashboard")
    log.info("[APP] 🚀 Démarrage de l’application (lifespan)")

    # --- Phase de démarrage ---
    yield  # ⏸️ Exécution normale de l’application

    # --- Phase d’arrêt ---
    log.info("[APP] 🧹 Signal de shutdown reçu — stop_event déclenché pour les WS")
    if hasattr(app.state, "stop_event"):
        app.state.stop_event.set()

    try:
        if hasattr(app.state, "collector"):
            app.state.collector.stop()
            log.info("[APP] ✅ LogCollector arrêté proprement (lifespan)")
    except Exception as e:
        log.error(f"[APP] Erreur lors de l’arrêt du collector: {e}")


# ------------------------------------------------------------------
# 🧩 Création et configuration principale de l’application
# ------------------------------------------------------------------
def create_app(cfg):
    """Crée et retourne une instance complète de l’application FastAPI."""
    log = logging.getLogger("igt.dashboard")

    meta = cfg.get("dashboard", {})
    app = FastAPI(
        title="Ultramotion Web Monitor",
        version="0.2-dev",
        description="Monitoring temps réel du pipeline Ultramotion",
        lifespan=lifespan,  # ✅ nouvelle gestion du shutdown
    )

    # ------------------------------------------------------------------
    # 🔒 Middleware de vérification du jeton HTTP (pas WebSocket)
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        if request.scope["type"] != "http":
            return await call_next(request)

        if meta.get("security", {}).get("enabled", False):
            expected = meta["security"].get("token")
            token = request.headers.get("X-API-Token")
            if token != expected:
                log.warning(f"Requête non autorisée depuis {request.client.host}")
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)

    # ------------------------------------------------------------------
    # 🌐 CORS + gestion d'erreurs globale
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # à restreindre plus tard si besoin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse({"error": "invalid request", "details": exc.errors()}, status_code=422)

    # ------------------------------------------------------------------
    # 📦 Montage des fichiers statiques
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[2]
    assets_path = base_dir / "assets"
    javascript_path = assets_path / "javascript"

    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")
    app.mount("/javascript", StaticFiles(directory=str(javascript_path)), name="javascript")

    app.state.cfg = cfg

    # ------------------------------------------------------------------
    # 🧩 Initialisation du Collector
    # ------------------------------------------------------------------
    try:
        # 1️⃣ Lecture des chemins depuis config YAML
        collector_cfg = meta.get("collector", {})
        pipeline_path = collector_cfg.get("pipeline_path")
        kpi_path = collector_cfg.get("kpi_path")

        # 2️⃣ Valeurs par défaut si non précisées
        if not pipeline_path or not kpi_path:
            log.warning("[APP] Chemins collector non définis — utilisation des logs réels du pipeline")

            # Remonte de 4 niveaux pour atteindre la racine du projet
            default_dir = Path(__file__).resolve().parents[4] / "logs"

            pipeline_path = str(default_dir / "pipeline.log")
            kpi_path = str(default_dir / "kpi.log")

            # 🔍 Ajout de logs et prints explicites pour vérification
            log.info(f"[APP] 🔍 Dossier racine logs détecté : {default_dir}")
            log.info(f"[APP] 🔗 pipeline.log  => {pipeline_path}")
            log.info(f"[APP] 🔗 kpi.log       => {kpi_path}")

        # 3️⃣ Instanciation du collector réel
        app.state.collector = Collector(pipeline_path=pipeline_path, kpi_path=kpi_path)
        app.state.collector.start()
        log.info(f"[APP] Collector réel attaché ✅ ({pipeline_path}, {kpi_path})")

    except Exception as e:
        # Fallback simulé si le collector échoue
        log.warning(f"[APP] Collector réel indisponible ({e}); utilisation du mock ⚠️")
        tmp1 = tempfile.NamedTemporaryFile(delete=False).name
        tmp2 = tempfile.NamedTemporaryFile(delete=False).name
        try:
            mock_collector = Collector(tmp1, tmp2)
            app.state.collector = mock_collector
            log.info("[APP] Collector simulé attaché à app.state.collector (fallback)")
        except Exception as e2:
            log.error(f"[APP] Impossible d’attacher un collector (réel ni simulé): {e2}")
            app.state.collector = None

    # ------------------------------------------------------------------
    # 🔌 Enregistrement des routes
    # ------------------------------------------------------------------
    register_api_routes(app)
    register_ws_routes(app)

    # ------------------------------------------------------------------
    # 💓 Health Check
    # ------------------------------------------------------------------
    @app.get("/api/health")
    async def health_check(request: Request):
        token = request.headers.get("X-API-Token")
        expected = meta.get("security", {}).get("token")
        if meta.get("security", {}).get("enabled", False) and token != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"status": "ok"}

    from sandbox.web_monitor.common.app.api_routes import router as api_router
    app.include_router(api_router)

    # ------------------------------------------------------------------
    # ✅ Gestion silencieuse du CancelledError pendant le shutdown
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def ignore_cancelled_error(request: Request, call_next):
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            log.info("[APP] ✅ Arrêt propre détecté (CancelledError ignoré)")
            return PlainTextResponse("Server shutting down...", status_code=503)

    log.info("Application FastAPI configurée avec CORS, sécurité et collector prêt ✅")
    return app
