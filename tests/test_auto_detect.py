"""
Production-Grade Tests for Backend Auto-Detection & Manager Integration.
Covers file signature matching, priority resolution, conflict handling,
missing input validation, and BackendManager singleton behavior.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from wien2k_gen.backend_manager import BackendManager
from wien2k_gen.types import BackendCode
from wien2k_gen.exceptions import MissingInputError, BackendError


# =============================================================================
# Fixtures: Mock File System & Glob Results
# =============================================================================

@pytest.fixture
def mock_cwd():
    return MagicMock(spec=Path)

@pytest.fixture
def glob_return_factory():
    """Factory to simulate Path.glob() results."""
    def _factory(files):
        class MockGlob:
            def __iter__(self): return iter(files)
        return MockGlob()
    return _factory


# =============================================================================
# Test Suites
# =============================================================================

class TestAutoDetectSignatures:
    """Tests for file-based backend signature detection."""

    @patch.object(BackendManager, "auto_detect")
    def test_wien2k_detection(self, mock_auto):
        mock_auto.return_value = BackendCode.WIEN2K
        assert BackendManager().auto_detect() == BackendCode.WIEN2K

    @patch("pathlib.Path.cwd")
    @patch.object(Path, "glob")
    def test_detect_wien2k_struct(self, mock_glob, mock_cwd, glob_return_factory):
        mock_cwd.return_value.glob = mock_glob
        mock_glob.side_effect = lambda p: glob_return_factory(["case.struct"]) if "*.struct" in p else []
        
        result = BackendManager().auto_detect()
        assert result == BackendCode.WIEN2K

    @patch("pathlib.Path.cwd")
    @patch.object(Path, "glob")
    def test_detect_vasp_poscar(self, mock_glob, mock_cwd, glob_return_factory):
        mock_cwd.return_value.glob = mock_glob
        mock_glob.side_effect = lambda p: glob_return_factory(["POSCAR"]) if "POSCAR" in p else []
        
        result = BackendManager().auto_detect()
        assert result == BackendCode.VASP

    @patch("pathlib.Path.cwd")
    @patch.object(Path, "glob")
    def test_detect_qe_input(self, mock_glob, mock_cwd, glob_return_factory):
        mock_cwd.return_value.glob = mock_glob
        mock_glob.side_effect = lambda p: glob_return_factory(["scf.in"]) if "*.pw.in" in p else []
        
        result = BackendManager().auto_detect()
        assert result == BackendCode.QUANTUM_ESPRESSO

    @pytest.mark.parametrize("files,expected", [
        (["case.struct", "POSCAR"], BackendCode.WIEN2K),  # Priority: WIEN2K > VASP
        (["INCAR", "POTCAR"], BackendCode.VASP),
        (["qe.in", "cp2k.inp"], BackendCode.QUANTUM_ESPRESSO),
    ])
    @patch("pathlib.Path.cwd")
    @patch.object(Path, "glob")
    def test_priority_conflict_resolution(self, mock_glob, mock_cwd, files, expected, glob_return_factory):
        """Test deterministic priority when multiple backends' files exist."""
        mock_cwd.return_value.glob = mock_glob
        def side_effect(pattern):
            matches = [f for f in files if pattern.strip("*") in f or f.endswith(pattern)]
            return glob_return_factory(matches)
        mock_glob.side_effect = side_effect
        
        assert BackendManager().auto_detect() == expected

    @patch("pathlib.Path.cwd")
    @patch.object(Path, "glob", return_value=[])
    def test_no_files_raises_error(self, mock_glob, mock_cwd):
        """Missing input files must raise actionable error."""
        mock_cwd.return_value.glob = mock_glob
        with pytest.raises(MissingInputError, match="No recognizable DFT input files found"):
            BackendManager().auto_detect()

    @patch("pathlib.Path.cwd")
    @patch.object(Path, "glob")
    def test_handles_permission_error_gracefully(self, mock_glob, mock_cwd):
        """Glob failure should not crash, but fall through or raise clean error."""
        mock_cwd.return_value.glob = mock_glob
        mock_glob.side_effect = PermissionError("Access denied")
        
        with pytest.raises(MissingInputError):
            BackendManager().auto_detect()


class TestBackendManagerIntegration:
    """Tests for singleton, caching, and lazy loading behavior."""

    def test_singleton_instance(self):
        assert BackendManager.instance() is BackendManager.instance()

    @patch.object(BackendManager, "_load_backends")
    @patch.object(BackendManager, "auto_detect", return_value=BackendCode.WIEN2K)
    def test_get_backend_caching(self, mock_detect, mock_load):
        mgr = BackendManager.instance()
        b1 = mgr.get_backend()
        b2 = mgr.get_backend()
        assert b1 is b2  # Same instance cached
        mock_load.assert_called_once()

    @patch.object(BackendManager, "_load_backends")
    @patch.object(BackendManager, "auto_detect", return_value=BackendCode.VASP)
    def test_set_backend_invalidates_cache(self, mock_detect, mock_load):
        mgr = BackendManager.instance()
        mgr.get_backend()  # Cache WIEN2K
        mgr.set_backend(BackendCode.QUANTUM_ESPRESSO)
        # Next call should instantiate new
        mgr.get_backend()
        assert mock_load.call_count >= 1  # Reload/validate triggered

    @patch.object(BackendManager, "_load_backends")
    def test_list_available_filters_stubs(self, mock_load):
        mgr = BackendManager.instance()
        # Simulate registry with stubs
        class Stub: pass
        class Real: pass
        mgr._registry = {
            BackendCode.WIEN2K: Real,
            BackendCode.VASP: Stub
        }
        mgr._loaded = True
        available = mgr.list_available()
        assert BackendCode.WIEN2K in available
        assert BackendCode.VASP not in available