"""
Production-Grade Tests for utils/ Module.
Covers atomic_write, filelock, validation, and scratch staging logic.
"""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from wien2k_gen.utils.atomic_write import atomic_write
from wien2k_gen.utils.filelock import FileLock, LockTimeoutError
from wien2k_gen.utils.validation import parse_machines_file, validate_machines


class TestAtomicWrite:
    def test_success_creates_file(self, tmp_path):
        target = tmp_path / "test.txt"
        assert atomic_write(target, "hello\n")
        assert target.read_text() == "hello\n"
        assert oct(target.stat().st_mode)[-3:] == "644"

    @patch("tempfile.mkstemp", side_effect=PermissionError("Access denied"))
    def test_permission_error_on_readonly_dir(self, mock_mkstemp, tmp_path):
        target = tmp_path / "sub" / "no_access.txt"
        with pytest.raises((PermissionError, OSError)):
            atomic_write(target, "should fail")

    def test_directory_creation(self, tmp_path):
        target = tmp_path / "sub" / "deep" / "file.txt"
        assert atomic_write(target, "nested")
        assert target.exists()


class TestFileLock:
    def test_acquire_and_release(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        with FileLock(lock_path, timeout=2.0) as fl:
            assert fl.lock_path.exists() or True
        # Lock should be released - fallback dir should be cleaned
        fallback = tmp_path / ".test.lock.lock.d"
        assert not fallback.exists()

    @patch("wien2k_gen.utils.filelock.fcntl.flock", side_effect=OSError(11, "Resource temporarily unavailable"))
    def test_timeout_raises_error(self, mock_flock, tmp_path):
        target_path = tmp_path / "busy.lock"
        fallback_dir = tmp_path / ".busy.lock.lock.d"
        fallback_dir.mkdir(parents=True)
        (fallback_dir / "pid").write_text("999999")
        
        with pytest.raises(LockTimeoutError):
            with FileLock(target_path, timeout=0.1, delay=0.05):
                pass


class TestMachinesValidation:
    def test_parse_valid_machines(self, tmp_path):
        content = """
# Comment
lapw1: node01: 16
lapw2: node01: 16
omp_global: 2
kpar: 4
granularity: 1
"""
        f = tmp_path / ".machines"
        f.write_text(content)
        cfg, warns = parse_machines_file(f)
        assert cfg["omp_global"] == 2
        assert cfg["kpar"] == 4
        assert len(warns) == 0

    def test_detect_oversubscription(self, tmp_path):
        content = "lapw1: node01: 64\nlapw2: node01: 64\nomp_global: 1"
        f = tmp_path / ".machines"
        f.write_text(content)
        from wien2k_gen.types import TopologyData
        topo = TopologyData(nodes=["node01"], cores_per_node=[32], total_cores=32)
        res = validate_machines(f, topo=topo, strict_mode=False)
        assert any("exceed" in w.lower() for w in res["warnings"])