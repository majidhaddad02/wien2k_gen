"""
Wien2kGen REST API & Web Dashboard subpackage.
Provides HTTP server, JSON endpoints, and a single-page monitoring dashboard.
"""

from .server import main, Wien2kAPIHandler

__all__ = ["main", "Wien2kAPIHandler"]
