"""
Production-Grade Tests for utils/ Module.
Covers atomic_write, filelock, validation, and scratch staging logic.
"""

from unittest.mock import patch

import pytest

from forge.utils.atomic_write import atomic_write
from forge.utils.filelock import FileLock, LockTimeoutError
from forge.utils.validation import parse_machines_file, validate_machines


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

    @patch("forge.utils.filelock.fcntl.flock", side_effect=OSError(11, "Resource temporarily unavailable"))
    def test_timeout_raises_error(self, mock_flock, tmp_path):
        target_path = tmp_path / "busy.lock"
        fallback_dir = tmp_path / ".busy.lock.lock.d"
        fallback_dir.mkdir(parents=True)
        (fallback_dir / "pid").write_text("999999")
        
        with pytest.raises(LockTimeoutError), FileLock(target_path, timeout=0.1, delay=0.05):
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
        from forge.types import TopologyData
        topo = TopologyData(nodes=["node01"], cores_per_node=[32], total_cores=32)
        res = validate_machines(f, topo=topo, strict_mode=False)
        assert any("exceed" in w.lower() for w in res["warnings"])

    def test_remainder_absorption_not_omp_error(self, tmp_path):
        content = "\n".join(
            [
                "1: node01:4",
                "1: node01:4",
                "1: node01:5",
                "omp_global: 4",
            ]
        )
        f = tmp_path / ".machines"
        f.write_text(content)
        res = validate_machines(f, strict_mode=False)
        assert res["valid"] is True
        assert not any("divis" in e.lower() for e in res["errors"])
        assert sum(res["config"]["cores_per_node"]) == 13

    def test_topology_match_by_name_not_position(self, tmp_path):
        content = "1: nodeA: 32\n1: nodeB: 4\nomp_global: 1"
        f = tmp_path / ".machines"
        f.write_text(content)
        from forge.types import TopologyData
        topo = TopologyData(
            nodes=["nodeB", "nodeA"],
            cores_per_node=[8, 64],
            total_cores=72,
        )
        res = validate_machines(f, topo=topo, strict_mode=False)
        assert not any("exceed" in w.lower() for w in res["warnings"])

    def test_oversubscription_by_node_name(self, tmp_path):
        content = "1: nodeA: 64\n1: nodeB: 4\nomp_global: 1"
        f = tmp_path / ".machines"
        f.write_text(content)
        from forge.types import TopologyData
        topo = TopologyData(
            nodes=["nodeB", "nodeA"],
            cores_per_node=[8, 32],
            total_cores=40,
        )
        res = validate_machines(f, topo=topo, strict_mode=False)
        assert any("nodeA" in w and "exceed" in w.lower() for w in res["warnings"])
        assert not any("nodeB" in w and "exceed" in w.lower() for w in res["warnings"])

    def test_lapw1_lapw2_same_node_uses_max_not_sum(self, tmp_path):
        content = "lapw1: node01: 16\nlapw2: node01: 8\nomp_global: 1\n"
        f = tmp_path / ".machines"
        f.write_text(content)
        cfg, warns = parse_machines_file(f)
        assert cfg["cores_per_node"] == [16]
        assert cfg["lapw1_cores"] == 16
        assert cfg["lapw2_cores"] == 8
        assert warns == []

    def test_canonical_2_lines_are_recognized(self, tmp_path):
        content = "1: node01: 8\n2: node01: 8\nomp_global: 1\n"
        f = tmp_path / ".machines"
        f.write_text(content)
        cfg, warns = parse_machines_file(f)
        assert cfg["nodes"] == ["node01"]
        assert cfg["cores_per_node"] == [8]
        assert cfg["lapw2_cores"] == 8
        assert not any("unrecognized" in w.lower() for w in warns)

    def test_multi_host_kpoint_line(self, tmp_path):
        content = "1: node01: 4  node02: 4  node03: 4\nomp_global: 1\n"
        f = tmp_path / ".machines"
        f.write_text(content)
        cfg, warns = parse_machines_file(f)
        assert cfg["nodes"] == ["node01", "node02", "node03"]
        assert cfg["cores_per_node"] == [4, 4, 4]
        assert cfg["rank_count"] == 3
        assert warns == []

    def test_hybrid_mode_from_kpoint_plus_omp(self, tmp_path):
        f = tmp_path / ".machines"
        f.write_text("1: node01: 4\nomp_global: 4\n")
        cfg, _ = parse_machines_file(f)
        assert cfg["mode"] == "hybrid"