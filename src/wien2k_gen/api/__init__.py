"""
Wien2kGen REST API & Web Dashboard subpackage.
Provides HTTP server, JSON endpoints, and a single-page monitoring dashboard.
"""

from .server import Wien2kAPIHandler, main

__all__ = ["Wien2kAPIHandler", "main"]
