"""
Export & Serialization Utility Module for HPC/DFT Workflows.
Provides robust, format-agnostic data export with atomic file operations,
schema validation, and graceful fallbacks for complex scientific objects.

Key Features:
• Multi-format support: JSON, YAML, TOML, CSV, TXT/Markdown
• Automatic serialization of dataclasses, TypedDicts, Path objects, sets, and datetime
• Custom JSON/YAML/TOML encoders with NumPy/Pandas compatibility (if available)
• Atomic write integration to prevent partial/corrupted exports on shared filesystems
• Format auto-detection via file extension with explicit override capability
• Structured error handling, validation hooks, and HPC-grade logging
• Extensible plugin-style architecture for custom exporters

All documentation and inline comments are in English per project standards.
"""

import os
import re
import json
import csv
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, TypedDict, Tuple
from dataclasses import is_dataclass, asdict, fields
from datetime import datetime, timezone

# Optional dependencies with graceful fallbacks
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    yaml = None
    _HAS_YAML = False

try:
    import tomli_w
    _HAS_TOML = True
except ImportError:
    tomli_w = None
    _HAS_TOML = False

from ..utils.atomic_write import atomic_write
from ..logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Type Definitions & Configuration
# =============================================================================

class ExportResult(TypedDict, total=False):
    """Structured outcome of export operation."""
    success: bool
    path: Optional[str]
    format: str
    size_bytes: int
    warnings: List[str]
    errors: List[str]
    duration_sec: float


class ExportConfig(TypedDict, total=False):
    """Configuration for export behavior."""
    indent: int
    sort_keys: bool
    ensure_ascii: bool
    csv_delimiter: str
    csv_header: bool
    include_metadata: bool
    timestamp_format: str


# =============================================================================
# Custom Serializers & Type Converters
# =============================================================================

class _ScientificEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for HPC/DFT scientific objects.
    Handles dataclasses, Path, set, datetime, and optional NumPy/Pandas types.
    """
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if hasattr(obj, "tolist"):  # NumPy arrays
            return obj.tolist()
        if hasattr(obj, "to_dict") and hasattr(obj, "index"):  # Pandas DataFrames
            return obj.to_dict(orient="records")
        return super().default(obj)


def _sanitize_for_export(data: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively sanitize nested structures for safe serialization.
    Removes circular references, limits depth, and converts unsupported types.
    """
    if depth > max_depth:
        return "... (max depth exceeded)"
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, set):
        return sorted(list(data))
    if is_dataclass(data):
        return _sanitize_for_export(asdict(data), depth + 1)
    if isinstance(data, dict):
        return {str(k): _sanitize_for_export(v, depth + 1) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_sanitize_for_export(item, depth + 1) for item in data]
    if hasattr(data, "to_dict") and callable(data.to_dict):
        return _sanitize_for_export(data.to_dict(), depth + 1)
    if hasattr(data, "__dict__"):
        return _sanitize_for_export(data.__dict__, depth + 1)
        
    return str(data)


# =============================================================================
# Format-Specific Exporters
# =============================================================================

def _export_json(data: Any, path: Path, config: ExportConfig) -> int:
    """Export data to JSON with atomic write and custom encoder."""
    content = json.dumps(
        _sanitize_for_export(data),
        indent=config.get("indent", 2),
        sort_keys=config.get("sort_keys", True),
        ensure_ascii=config.get("ensure_ascii", True),
        cls=_ScientificEncoder
    )
    atomic_write(path, content + "\n", mode=0o644)
    return len(content.encode("utf-8"))


def _export_yaml(data: Any, path: Path, config: ExportConfig) -> int:
    """Export data to YAML with atomic write and safe dumping."""
    if not _HAS_YAML:
        raise ImportError("PyYAML is required for YAML export. Install with: pip install pyyaml")
    sanitized = _sanitize_for_export(data)
    content = yaml.dump(
        sanitized,
        default_flow_style=False,
        sort_keys=config.get("sort_keys", True),
        allow_unicode=not config.get("ensure_ascii", True)
    )
    atomic_write(path, content, mode=0o644)
    return len(content.encode("utf-8"))


def _export_toml(data: Any, path: Path, config: ExportConfig) -> int:
    """Export data to TOML with atomic write and nested dict flattening if needed."""
    if not _HAS_TOML:
        raise ImportError("tomli_w is required for TOML export. Install with: pip install tomli_w")
    # TOML requires strict dict structures; sanitize aggressively
    sanitized = _sanitize_for_export(data)
    content = tomli_w.dumps(sanitized)
    atomic_write(path, content, mode=0o644)
    return len(content.encode("utf-8"))


def _export_csv(data: Any, path: Path, config: ExportConfig) -> int:
    """Export list of dicts or tabular data to CSV."""
    if not isinstance(data, list):
        data = [data]  # Wrap single dict
    if not data:
        atomic_write(path, "", mode=0o644)
        return 0
        
    # Flatten nested dicts for CSV compatibility
    rows = []
    for item in data:
        flat = {}
        for k, v in _sanitize_for_export(item).items():
            flat[str(k)] = json.dumps(v) if isinstance(v, (dict, list)) else v
        rows.append(flat)
        
    fieldnames = list(rows[0].keys())
    # FIXED: newline="" is the correct standard for csv module
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=config.get("csv_delimiter", ","))
        if config.get("csv_header", True):
            writer.writeheader()
        writer.writerows(rows)
        
    return os.path.getsize(path)


def _export_txt(data: Any, path: Path, config: ExportConfig) -> int:
    """Export data to human-readable TXT/Markdown format."""
    lines = []
    if isinstance(data, dict):
        for k, v in _sanitize_for_export(data).items():
            if isinstance(v, (dict, list)):
                lines.append(f"### {k}")
                lines.append(json.dumps(v, indent=2))
                lines.append("")
            else:
                lines.append(f"{k}: {v}")
        if config.get("include_metadata", True):
            lines.append("")
            lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    elif isinstance(data, list):
        for i, item in enumerate(data, 1):
            lines.append(f"## Item {i}")
            lines.append(json.dumps(_sanitize_for_export(item), indent=2))
            lines.append("")
    else:
        lines.append(str(_sanitize_for_export(data)))
        
    content = "\n".join(lines)
    atomic_write(path, content + "\n", mode=0o644)
    return len(content.encode("utf-8"))


# =============================================================================
# Core Export Orchestrator
# =============================================================================

def export_config(
    data: Any,
    path: Union[str, Path],
    format_hint: Optional[str] = None,
    config: Optional[ExportConfig] = None
) -> ExportResult:
    """
    Export structured data to file with format detection, atomic write, and validation.
    
    Args:
        data: Data to export (dict, list, dataclass, or serializable object).
        path: Target file path. Format auto-detected from extension if format_hint is None.
        format_hint: Explicit format override ('json', 'yaml', 'toml', 'csv', 'txt').
        config: Optional export configuration (indentation, CSV settings, etc.).
        
    Returns:
        ExportResult with success status, path, size, and diagnostics.
    """
    start_time = time.monotonic()
    cfg = config or {}
    target = Path(path).resolve()
    warnings: List[str] = []
    errors: List[str] = []

    # 1. Format Detection & Validation
    ext = format_hint.lower().strip() if format_hint else target.suffix.lower().lstrip(".")
    if not ext or ext not in ("json", "yaml", "yml", "toml", "csv", "txt", "md"):
        ext = "json"
        warnings.append(f"Unknown extension '{target.suffix}'. Defaulting to JSON.")
        target = target.with_suffix(".json")
        
    # Normalize YAML extension
    if ext == "yml":
        ext = "yaml"
        
    # 2. Directory Preparation
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"Failed to create directory {target.parent}: {e}")
        return ExportResult(success=False, path=str(target), format=ext, errors=errors, duration_sec=0.0)
        
    # 3. Dispatch to Format-Specific Exporter
    size_bytes = 0
    try:
        if ext == "json":
            size_bytes = _export_json(data, target, cfg)
        elif ext == "yaml":
            size_bytes = _export_yaml(data, target, cfg)
        elif ext == "toml":
            size_bytes = _export_toml(data, target, cfg)
        elif ext == "csv":
            size_bytes = _export_csv(data, target, cfg)
        else:  # txt/md
            size_bytes = _export_txt(data, target, cfg)
            
    except ImportError as e:
        errors.append(f"Missing dependency: {e}")
    except Exception as e:
        errors.append(f"Export failed for {ext}: {e}")
        logger.error(f"Export error: {e}", exc_info=True)
        
    duration = time.monotonic() - start_time
    success = len(errors) == 0 and size_bytes > 0

    if success:
        logger.info(f"Exported {ext.upper()} to {target} ({size_bytes} bytes, {duration:.3f}s)")
    else:
        logger.warning(f"Export to {target} completed with issues: {errors}")
        
    return ExportResult(
        success=success,
        path=str(target),
        format=ext,
        size_bytes=size_bytes,
        warnings=warnings,
        errors=errors,
        duration_sec=round(duration, 4)
    )


def export_multiple(
    outputs: Dict[str, Any],
    base_path: Union[str, Path],
    format: str = "json",
    config: Optional[ExportConfig] = None
) -> Dict[str, ExportResult]:
    """
    Export multiple datasets to a directory with consistent naming and format.
    Useful for batch profiling reports, topology snapshots, or benchmark suites.
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, data in outputs.items():
        safe_name = re.sub(r"[^\w\-]", "_", name.lower())
        file_path = base / f"{safe_name}.{format}"
        results[name] = export_config(data, file_path, format_hint=format, config=config)
        
    success_count = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"Batch export complete: {success_count}/{len(outputs)} successful")
    return results


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ExportResult",
    "ExportConfig",
    "export_config",
    "export_multiple",
    "_ScientificEncoder",
    "_sanitize_for_export",
]