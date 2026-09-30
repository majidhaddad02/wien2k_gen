"""Tests for forge.core.builder.build_auto suggestion-validation flow."""

from unittest.mock import MagicMock, patch

from forge.backends.base import ValidationIssue
from forge.backends.wien2k.core import Wien2kBackend
from forge.core.builder import BuildResult, build_auto
from forge.core.topology import Topology


def _topo():
    return Topology(nodes=["n01"], cores_per_node=[8], env_type="local")


@patch("forge.core.builder._get_current_backend")
def test_zero_cores_blocks_build_auto(_get_backend):
    backend = Wien2kBackend()
    _get_backend.return_value = backend
    with (
        patch.object(backend, "validate_suggestion", wraps=backend.validate_suggestion),
        patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None),
        patch("forge.core.hardware.check_elpa_available", return_value=False),
    ):
        result = build_auto(
            _topo(),
            suggestion={"mode": "mpi", "recommended_total_cores": 0},
            backup=False,
            dry_run=True,
        )
    assert isinstance(result, BuildResult)
    assert result.success is False
    assert result.error_message
    assert "must be > 0" in result.error_message
    assert not any("must be > 0" in w for w in result.warnings)


@patch("forge.core.builder._get_current_backend")
def test_validate_suggestion_exception_fails_build(_get_backend):
    backend = MagicMock()
    backend.__class__.__name__ = "FakeBackend"
    backend.validate_suggestion.side_effect = RuntimeError("validator crashed")
    _get_backend.return_value = backend
    result = build_auto(
        _topo(),
        suggestion={"mode": "mpi", "recommended_total_cores": 8},
        backup=False,
        dry_run=True,
    )
    assert isinstance(result, BuildResult)
    assert result.success is False
    assert "validator crashed" in (result.error_message or "")


@patch("forge.core.builder._get_current_backend")
def test_validation_warnings_do_not_block(_get_backend):
    backend = MagicMock()
    backend.__class__.__name__ = "FakeBackend"
    backend.validate_suggestion.return_value = [
        ValidationIssue("warning", "advisory only"),
    ]
    backend.generate_input.return_value = "lapw0: n01:1\n"
    _get_backend.return_value = backend
    result = build_auto(
        _topo(),
        suggestion={"mode": "mpi", "recommended_total_cores": 8},
        backup=False,
        dry_run=True,
    )
    assert result.success is True
    assert "advisory only" in result.warnings
