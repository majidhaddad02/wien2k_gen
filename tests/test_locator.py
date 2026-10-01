"""Tests for forge.core.locator.find_elpa_dir layered ELPA detection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from forge.backends.elpa_selector import _resolve_elpa_dir
from forge.core.locator import find_elpa_dir


def _fake_elpa_prefix(tmp_path: Path, libdir: str = "lib", libname: str = "libelpa.so") -> Path:
    lib = tmp_path / libdir
    lib.mkdir(parents=True, exist_ok=True)
    (lib / libname).write_bytes(b"")
    return tmp_path


def _clear_elpa_signals(monkeypatch):
    for var in (
        "ELPA_DIR",
        "ELPA_HOME",
        "ELPA_ROOT",
        "EBROOTELPA",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "MODULEPATH",
    ):
        monkeypatch.delenv(var, raising=False)


class TestFindElpaDirEnvVars:
    def test_elpa_dir_with_libelpa_so(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path)
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("ELPA_DIR", str(prefix))
        assert find_elpa_dir() == str(prefix.resolve())

    def test_elpa_home_when_elpa_dir_unset(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path)
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("ELPA_HOME", str(prefix))
        assert find_elpa_dir() == str(prefix.resolve())

    def test_elpa_root_when_dir_and_home_unset(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path)
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("ELPA_ROOT", str(prefix))
        assert find_elpa_dir() == str(prefix.resolve())

    def test_env_var_without_libelpa_is_rejected(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("ELPA_DIR", str(empty))
        with patch("forge.core.locator._from_pkg_config", return_value=None):
            assert find_elpa_dir() is None


class TestFindElpaDirEasyBuild:
    def test_ebrootelpa(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path)
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("EBROOTELPA", str(prefix))
        assert find_elpa_dir() == str(prefix.resolve())


class TestFindElpaDirPkgConfig:
    def test_pkg_config_elpa_prefix(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path)
        _clear_elpa_signals(monkeypatch)

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if cmd[:2] == ["pkg-config", "--list-all"]:
                result.stdout = "elpa_openmp          ELPA OpenMP\nopenssl              OpenSSL\n"
                result.stderr = ""
                return result
            if cmd[:2] == ["pkg-config", "--variable=prefix"]:
                result.stdout = str(prefix) + "\n"
                result.stderr = ""
                return result
            raise AssertionError(f"unexpected command: {cmd}")

        with patch("forge.core.locator.subprocess.run", side_effect=fake_run):
            assert find_elpa_dir() == str(prefix.resolve())

    def test_pkg_config_missing_falls_through(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path, libdir="lib64", libname="libelpa.so")
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("LD_LIBRARY_PATH", str(prefix / "lib64"))
        with patch(
            "forge.core.locator.subprocess.run",
            side_effect=FileNotFoundError("pkg-config"),
        ):
            assert find_elpa_dir() == str(prefix.resolve())


class TestFindElpaDirLinkerPath:
    def test_ld_library_path_lib64(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path, libdir="lib64", libname="libelpa_openmp.so")
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("LD_LIBRARY_PATH", str(prefix / "lib64"))
        with patch("forge.core.locator._from_pkg_config", return_value=None):
            assert find_elpa_dir() == str(prefix.resolve())

    def test_library_path_when_ld_library_path_empty(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path, libdir="lib", libname="libelpa.a")
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("LIBRARY_PATH", str(prefix / "lib"))
        with patch("forge.core.locator._from_pkg_config", return_value=None):
            assert find_elpa_dir() == str(prefix.resolve())


class TestFindElpaDirFallbackAndPriority:
    def test_nothing_returns_none(self, monkeypatch):
        _clear_elpa_signals(monkeypatch)
        with patch("forge.core.locator._from_pkg_config", return_value=None):
            assert find_elpa_dir() is None

    def test_env_var_wins_over_pkg_config(self, tmp_path, monkeypatch):
        env_prefix = _fake_elpa_prefix(tmp_path / "from_env")
        pc_prefix = _fake_elpa_prefix(tmp_path / "from_pc")
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("ELPA_DIR", str(env_prefix))

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if cmd[:2] == ["pkg-config", "--list-all"]:
                result.stdout = "elpa                 ELPA\n"
                result.stderr = ""
                return result
            if cmd[:2] == ["pkg-config", "--variable=prefix"]:
                result.stdout = str(pc_prefix) + "\n"
                result.stderr = ""
                return result
            raise AssertionError(f"unexpected command: {cmd}")

        with patch("forge.core.locator.subprocess.run", side_effect=fake_run):
            assert find_elpa_dir() == str(env_prefix.resolve())


class TestResolveElpaDirWrapper:
    def test_wrapper_delegates_to_find_elpa_dir(self, tmp_path, monkeypatch):
        prefix = _fake_elpa_prefix(tmp_path)
        _clear_elpa_signals(monkeypatch)
        monkeypatch.setenv("ELPA_DIR", str(prefix))
        assert _resolve_elpa_dir() == str(prefix.resolve())

    def test_wrapper_returns_empty_string_when_missing(self, monkeypatch):
        _clear_elpa_signals(monkeypatch)
        with patch("forge.core.locator.find_elpa_dir", return_value=None):
            assert _resolve_elpa_dir() == ""
