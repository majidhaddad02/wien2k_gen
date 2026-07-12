"""
Lightweight REST API Server for Wien2kGen.
Uses Python stdlib http.server only -- no external web framework.

Endpoints:
  GET  /api/v1/status             -> project version, uptime, active workflows
  GET  /api/v1/topology           -> hardware topology JSON
  POST /api/v1/optimize           -> accepts problem params, returns ResourceSuggestion
  GET  /api/v1/jobs               -> list submitted jobs
  GET  /api/v1/jobs/{job_id}      -> job details
  POST /api/v1/submit             -> submit a job
  GET  /api/v1/convergence/{id}   -> convergence data
  GET  /api/v1/health             -> memory, disk, cpu usage
"""

import json
import logging
import os
import signal
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] api: %(message)s")
logger = logging.getLogger("wien2k_api")

START_TIME = time.time()

_job_store: dict[str, dict[str, Any]] = {}
_workflows: dict[str, dict[str, Any]] = {}
_log_entries: list[dict[str, Any]] = []

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_API_TOKEN = os.environ.get("WIEN2K_API_TOKEN", "")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    _cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(body)


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")


def _auth_check(handler: BaseHTTPRequestHandler) -> bool:
    if not _API_TOKEN:
        return True
    token = handler.headers.get("X-API-Key", "")
    return token == _API_TOKEN


def _parse_path(path: str) -> tuple[str, Optional[str]]:
    parsed = urlparse(path)
    return parsed.path, parsed.query


def _read_body(handler: BaseHTTPRequestHandler) -> Optional[dict[str, Any]]:
    try:
        length = int(handler.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = handler.rfile.read(length)
        return json.loads(raw)
    except Exception:
        return None


def _get_topology() -> dict[str, Any]:
    try:
        from ..core.scheduler import detect
        topo = detect()
        return topo.to_dict()
    except Exception:
        return _get_fallback_topology()


def _get_fallback_topology() -> dict[str, Any]:
    try:
        cpu_count = os.cpu_count() or 1
        return {
            "nodes": [os.uname().nodename],
            "cores_per_node": [cpu_count],
            "env_type": "local",
            "total_cores": cpu_count,
            "scheduler_hints": {},
            "heterogeneous": False,
            "memory_per_node": [0],
        }
    except Exception:
        return {
            "nodes": ["localhost"],
            "cores_per_node": [1],
            "env_type": "local",
            "total_cores": 1,
            "scheduler_hints": {},
            "heterogeneous": False,
            "memory_per_node": [0],
        }


def _get_health() -> dict[str, Any]:
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total_gb": round(mem.total / (1024**3), 1),
            "memory_used_gb": round(mem.used / (1024**3), 1),
            "memory_percent": mem.percent,
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "disk_used_gb": round(disk.used / (1024**3), 1),
            "disk_percent": disk.percent,
            "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0],
            "uptime_seconds": time.time() - START_TIME,
        }
    return {
        "cpu_percent": 0,
        "cpu_count": os.cpu_count() or 1,
        "memory_total_gb": 0,
        "memory_used_gb": 0,
        "memory_percent": 0,
        "disk_total_gb": 0,
        "disk_used_gb": 0,
        "disk_percent": 0,
        "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0],
        "uptime_seconds": time.time() - START_TIME,
    }


def _get_active_workflows_count() -> int:
    return sum(1 for w in _workflows.values() if w.get("status") in ("pending", "running"))


def _log(level: str, message: str) -> None:
    entry = {"timestamp": time.time(), "level": level, "message": message}
    _log_entries.append(entry)
    if len(_log_entries) > 500:
        _log_entries[:] = _log_entries[-500:]


# ---------------------------------------------------------------------------
# request handler
# ---------------------------------------------------------------------------

class Wien2kAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info(fmt % args)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _cors_headers(self)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: C901
        if not _auth_check(self):
            _json_response(self, {"error": "Unauthorized"}, 401)
            return
        path, _ = _parse_path(self.path)
        _log("INFO", f"GET {path}")

        # route GET
        if path == "/api/v1/status":
            self._handle_status()
        elif path == "/api/v1/topology":
            self._handle_topology()
        elif path == "/api/v1/health":
            self._handle_health()
        elif path == "/api/v1/jobs":
            self._handle_list_jobs()
        elif path.startswith("/api/v1/jobs/"):
            self._handle_job_detail(path)
        elif path == "/api/v1/workflows":
            self._handle_list_workflows()
        elif path.startswith("/api/v1/convergence/"):
            self._handle_convergence(path)
        elif path == "/api/v1/log" or path == "/api/v1/logs":
            self._handle_logs()
        elif path == "/" or path == "/index.html":
            self._handle_dashboard()
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if not _auth_check(self):
            _json_response(self, {"error": "Unauthorized"}, 401)
            return
        path, _ = _parse_path(self.path)
        _log("INFO", f"POST {path}")

        if path == "/api/v1/optimize":
            self._handle_optimize()
        elif path == "/api/v1/submit":
            self._handle_submit()
        else:
            _json_response(self, {"error": "Not found"}, 404)

    # -- GET handlers --

    def _handle_status(self) -> None:
        from .. import __version__
        _json_response(self, {
            "version": __version__,
            "uptime_seconds": time.time() - START_TIME,
            "active_workflows": _get_active_workflows_count(),
            "job_count": len(_job_store),
            "workflow_count": len(_workflows),
        })

    def _handle_topology(self) -> None:
        _json_response(self, _get_topology())

    def _handle_health(self) -> None:
        _json_response(self, _get_health())

    def _handle_list_jobs(self) -> None:
        jobs = list(_job_store.values())
        _json_response(self, {"jobs": jobs, "count": len(jobs)})

    def _handle_job_detail(self, path: str) -> None:
        parts = path.split("/")
        job_id = parts[-1] if parts else ""
        if not job_id or job_id == "jobs":
            _json_response(self, {"error": "Missing job_id"}, 400)
            return
        job = _job_store.get(job_id)
        if job is None:
            _json_response(self, {"error": "Job not found"}, 404)
            return
        _json_response(self, job)

    def _handle_list_workflows(self) -> None:
        wfs = list(_workflows.values())
        _json_response(self, {"workflows": wfs, "count": len(wfs)})

    def _handle_convergence(self, path: str) -> None:
        parts = path.split("/")
        wf_id = parts[-1] if parts else ""
        if not wf_id or wf_id == "convergence":
            _json_response(self, {"error": "Missing workflow_id"}, 400)
            return
        wf = _workflows.get(wf_id)
        if wf is None:
            _json_response(self, {"error": "Workflow not found"}, 404)
            return

        # generate pseudo-convergence data from workflow metadata
        try:
            from ..optimizer.convergence import generate_convergence_report
            results = wf.get("convergence_data", [])
            _json_response(self, {
                "workflow_id": wf_id,
                "results": results,
                "report": generate_convergence_report({"results": results}),
            })
        except Exception:
            _json_response(self, {
                "workflow_id": wf_id,
                "results": wf.get("convergence_data", []),
                "report": "No convergence data available",
            })

    def _handle_logs(self) -> None:
        limit = 100
        entries = _log_entries[-limit:]
        _json_response(self, {"entries": entries, "count": len(entries)})

    def _handle_dashboard(self) -> None:
        dash_path = Path(__file__).parent / "dashboard.html"
        if dash_path.exists():
            content = dash_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            _cors_headers(self)
            self.end_headers()
            self.wfile.write(content)
        else:
            _json_response(self, {"error": "Dashboard not found"}, 404)

    # -- POST handlers --

    def _handle_optimize(self) -> None:
        body = _read_body(self)
        if body is None:
            _json_response(self, {"error": "Invalid JSON body"}, 400)
            return

        try:
            from ..core.scheduler import detect
            from ..optimizer.advisor import suggest_optimal_resources
            topo = detect()
            suggestion = suggest_optimal_resources(topo, user_max_cores=body.get("max_cores"))
            _json_response(self, suggestion.to_dict())
        except Exception as exc:
            _json_response(self, {"error": str(exc), "mode": "hybrid", "recommended_total_cores": 1, "omp_threads_per_rank": 1, "reason": f"Fallback due to: {exc}"}, 200)

    def _handle_submit(self) -> None:
        body = _read_body(self)
        if body is None:
            _json_response(self, {"error": "Invalid JSON body"}, 400)
            return

        job_id = str(uuid.uuid4())[:12]
        now = time.time()
        job = {
            "job_id": job_id,
            "status": "pending",
            "submitted_at": now,
            "params": body,
            "logs": [],
        }
        _job_store[job_id] = job

        wf_id = f"wf-{job_id}"
        _workflows[wf_id] = {
            "workflow_id": wf_id,
            "job_id": job_id,
            "status": "pending",
            "created_at": now,
            "params": body,
            "convergence_data": [],
        }

        # simulate job start (in production this would submit to SLURM/DRMAA, etc.)
        _log("INFO", f"Job {job_id} submitted for backend={body.get('backend', 'wien2k')}")
        job["status"] = "running"
        _workflows[wf_id]["status"] = "running"

        _json_response(self, {"job_id": job_id, "workflow_id": wf_id, "status": "running"}, 201)


# ---------------------------------------------------------------------------
# server runner
# ---------------------------------------------------------------------------

def main(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), Wien2kAPIHandler)
    _log("INFO", f"Wien2kGen API server starting on port {port}")

    def _shutdown(signum: int, frame: Any) -> None:
        _log("INFO", f"Received signal {signum}, shutting down")
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"Serving on http://0.0.0.0:{port}")
    print(f"Wien2kGen API server listening on http://0.0.0.0:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _log("INFO", "Server stopped")


if __name__ == "__main__":
    port = int(os.environ.get("WIEN2K_API_PORT", 8080))
    main(port)
