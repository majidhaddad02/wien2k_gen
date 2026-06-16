"""
Production-Grade Tests for config.py Module.
Covers validation, environment precedence, file persistence, thread-safety,
and edge-case handling for AppConfig & ConfigManager.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from wien2k_gen.config import AppConfig, ConfigManager, load_config, validate_config, ensure_dirs
from wien2k_gen.exceptions import ConfigurationError


class TestAppConfigValidation:
    @pytest.mark.parametrize("max_cores,valid", [(0, False), (-5, False), (1, True), (64, True)])
    def test_max_cores_bounds(self, max_cores, valid):
        cfg = AppConfig(max_cores=max_cores)
        errors = ConfigManager()._validate_config(cfg)
        has_error = any("max_cores" in e for e in errors)
        assert (not valid) == has_error

    def test_log_level_enum_validation(self):
        cfg = AppConfig(log_level="TRACE")
        errors = ConfigManager()._validate_config(cfg)
        assert any("Invalid log_level" in e for e in errors)

    def test_backend_enum_validation(self):
        cfg = AppConfig(backend="gaussian")
        errors = ConfigManager()._validate_config(cfg)
        assert any("Unsupported backend" in e for e in errors)


class TestConfigManagerLifecycle:
    def test_singleton_consistency(self):
        m1 = ConfigManager.instance()
        m2 = ConfigManager.instance()
        assert m1 is m2

    def test_thread_safe_concurrent_load(self, temp_config_dir, clean_env, mock_hardware_profile):
        import threading
        results, errors = [], []
        
        def load_worker():
            try:
                cfg = load_config(file_path=temp_config_dir / "cfg.json", cli_override={"backend": "wien2k"})
                results.append(cfg)
            except Exception as e:
                errors.append(e)
                
        threads = [threading.Thread(target=load_worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        assert len(errors) == 0
        assert all(r.backend == "wien2k" for r in results)

    @patch("wien2k_gen.config.os.access", return_value=False)
    def test_scratch_permission_error(self, mock_access, temp_config_dir):
        with patch.dict(os.environ, {"SCRATCH": str(temp_config_dir / "readonly")}):
            # Simulate non-writable scratch
            cfg = load_config()
            errors = validate_config()
            assert any("not writable" in e.lower() for e in errors)


class ConfigPersistenceTest:
    def test_save_and_load_roundtrip(self, tmp_path, clean_env):
        cfg_path = tmp_path / "test_cfg.json"
        original = AppConfig(wienroot="/opt/test/WIEN2k", scratch_dir=str(tmp_path / "scratch"))
        ConfigManager.instance()._config = original
        ConfigManager.instance().save(cfg_path)
        
        # Verify file content
        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text())
        assert data["wienroot"] == "/opt/test/WIEN2k"
        
        # Reload
        reloaded = load_config(file_path=cfg_path)
        assert reloaded.wienroot == original.wienroot