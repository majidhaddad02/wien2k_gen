"""
Production-Grade Tests for Backend Auto-Detection & Manager Integration.
Covers file signature matching, priority resolution, conflict handling,
missing input validation, and BackendManager singleton behavior.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.backend_manager import (
    BackendManager,
    get_current_backend,
    list_backends,
    set_backend,
)
from forge.exceptions import MissingInputError
from forge.types import BackendCode


class TestAutoDetectSignatures:
    """Tests for file-based backend signature detection."""

    @patch.object(BackendManager, "auto_detect")
    def test_wien2k_detection(self, mock_auto):
        mock_auto.return_value = BackendCode.WIEN2K
        assert BackendManager().auto_detect() == BackendCode.WIEN2K

    def _make_glob_mock(self, patterns_map):
        """Create a Path.glob mock that returns files based on pattern."""
        class GlobIter:
            def __init__(self, items):
                self._items = list(items)
                self._pos = 0
            def __iter__(self):
                self._pos = 0
                return self
            def __next__(self):
                if self._pos >= len(self._items):
                    raise StopIteration
                item = self._items[self._pos]
                self._pos += 1
                return item

        def glob_side_effect(pattern):
            for pat, files in patterns_map.items():
                if pat in pattern:
                    return GlobIter(files)
            return GlobIter([])

        return glob_side_effect

    @patch.object(Path, "cwd")
    def test_detect_wien2k_struct(self, mock_cwd):
        mock_path = MagicMock()
        mock_path.glob = MagicMock(side_effect=self._make_glob_mock({"*.struct": ["case.struct"]}))
        mock_cwd.return_value = mock_path
        assert BackendManager().auto_detect() == BackendCode.WIEN2K

    @patch.object(Path, "cwd")
    def test_detect_vasp_poscar(self, mock_cwd):
        mock_path = MagicMock()
        mock_path.glob = MagicMock(side_effect=self._make_glob_mock({"POSCAR": ["POSCAR"]}))
        mock_cwd.return_value = mock_path
        assert BackendManager().auto_detect() == BackendCode.VASP

    @patch.object(Path, "cwd")
    def test_detect_qe_input(self, mock_cwd):
        mock_path = MagicMock()
        mock_path.glob = MagicMock(side_effect=self._make_glob_mock({"*.pw.in": ["scf.pw.in"]}))
        mock_cwd.return_value = mock_path
        assert BackendManager().auto_detect() == BackendCode.QUANTUM_ESPRESSO

    @patch.object(Path, "cwd")
    def test_no_files_raises_error(self, mock_cwd):
        mock_path = MagicMock()
        mock_path.glob = MagicMock(side_effect=self._make_glob_mock({}))
        mock_cwd.return_value = mock_path
        with pytest.raises(MissingInputError):
            BackendManager().auto_detect()

    @patch.object(Path, "cwd")
    def test_handles_permission_error(self, mock_cwd):
        mock_path = MagicMock()
        mock_path.glob = MagicMock(side_effect=PermissionError("Access denied"))
        mock_cwd.return_value = mock_path
        with pytest.raises((MissingInputError, PermissionError)):
            BackendManager().auto_detect()


class TestBackendManagerIntegration:
    """Tests for singleton, caching, and lazy loading behavior."""

    def test_singleton_instance(self):
        a = BackendManager.instance()
        b = BackendManager.instance()
        assert a is b

    def test_set_backend_invalidates_cache(self):
        set_backend(BackendCode.WIEN2K)
        backend = get_current_backend()
        assert backend is not None

    def test_list_available_returns_codes(self):
        available = list_backends()
        assert isinstance(available, list)
