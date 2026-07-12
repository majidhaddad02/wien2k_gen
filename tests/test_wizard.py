"""
Tests for wizard.py — WIENROOT detection, validation, and scratch health.
"""

import os
from unittest.mock import patch

from wien2k_gen.wizard import check_scratch_health, detect_wienroot_candidates, validate_wienroot


class TestDetectWienrootCandidates:
    def test_from_environment_variable(self):
        with patch.dict(os.environ, {"WIENROOT": "/opt/codes/WIEN2k_23"}):
            result = detect_wienroot_candidates()
            assert "/opt/codes/WIEN2k_23" in result

    def test_no_environment_variable(self):
        with patch.dict(os.environ, clear=True):
            result = detect_wienroot_candidates()
            assert "/opt/codes/WIEN2k_23" not in result

    def test_deduplication(self):
        with patch.dict(os.environ, {"WIENROOT": "/opt/codes/WIEN2k"}), patch("pathlib.Path.exists", return_value=True):
            result = detect_wienroot_candidates()
            assert result.count("/opt/codes/WIEN2k") == 1


class TestValidateWienroot:
    def test_valid_with_run_lapw(self, tmp_path):
        (tmp_path / "run_lapw").write_text("#!/bin/bash")
        assert validate_wienroot(str(tmp_path)) is True

    def test_valid_with_siteconfig(self, tmp_path):
        (tmp_path / "siteconfig_lapw").write_text("")
        assert validate_wienroot(str(tmp_path)) is True

    def test_neither_binary_present(self, tmp_path):
        assert validate_wienroot(str(tmp_path)) is False

    def test_not_a_directory(self, tmp_path):
        non_dir = tmp_path / "not_a_dir.txt"
        non_dir.write_text("hello")
        assert validate_wienroot(str(non_dir)) is False

    def test_nonexistent_path(self, tmp_path):
        assert validate_wienroot(str(tmp_path / "nonexistent")) is False


class TestCheckScratchHealth:
    def test_valid_writable_scratch(self, tmp_path):
        result = check_scratch_health(str(tmp_path))
        assert result["valid"] is True
        assert result["writable"] is True
        assert result["exists"] is True

    def test_nonexistent_cannot_create(self, tmp_path):
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            result = check_scratch_health(str(tmp_path / "nonexistent"))
            assert result["valid"] is False
            assert "Cannot create directory" in result["warning"]

    def test_readonly_scratch(self, tmp_path):
        p = tmp_path / "readonly"
        p.mkdir()
        with patch("os.access", return_value=False):
            result = check_scratch_health(str(p))
            assert result["writable"] is False

    def test_fs_type_detection(self, tmp_path):
        result = check_scratch_health(str(tmp_path))
        assert "fs_type" in result
