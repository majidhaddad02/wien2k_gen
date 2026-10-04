"""Tests for SLURM memory-unit parsing and mail-directive defaults."""

from __future__ import annotations

import pytest

from forge.core.topology import Topology
from forge.submit.slurm import (
    SlurmDirectives,
    SlurmJobSpec,
    _check_scheduler_limits,
    _format_sbatch_directives,
    generate_sbatch_script,
)


def _topo() -> Topology:
    return Topology(nodes=["n1"], cores_per_node=[8], env_type="slurm")


def _spec(**directive_kw) -> SlurmJobSpec:
    return SlurmJobSpec(
        topo=_topo(),
        exec_command="run_lapw -p",
        directives=SlurmDirectives(**directive_kw),
        validate_constraints=True,
    )


MEM_EXCEEDS = "exceeds job limit"


@pytest.mark.parametrize(
    "mem,limit_mb,should_warn",
    [
        ("64G", 65535, True),
        ("64g", 65535, True),
        ("64G", 65536, False),
        ("64g", 65536, False),
        ("2T", 2097151, True),
        ("2T", 2097152, False),
        ("500000K", 487, True),
        ("500000K", 488, False),
        ("1024", 1023, True),
        ("1024", 1024, False),
    ],
)
def test_memory_unit_conversion_against_job_limit(mem, limit_mb, should_warn, monkeypatch):
    monkeypatch.setattr("forge.submit.slurm.get_job_memory_limit_mb", lambda: limit_mb)
    warnings = _check_scheduler_limits(_spec(mem_per_node=mem, ntasks=1, cpus_per_task=1))
    flagged = any(MEM_EXCEEDS in w for w in warnings)
    assert flagged is should_warn


def test_memory_conversion_matrix_true_mb_values(monkeypatch):
    """Document converted MB values that the old uppercase-G-only check got wrong."""
    cases = {
        "64G": 65536,
        "64g": 65536,
        "2T": 2097152,
        "500000K": 488,
        "1024": 1024,
    }
    for mem, expected_mb in cases.items():
        monkeypatch.setattr(
            "forge.submit.slurm.get_job_memory_limit_mb",
            lambda mb=expected_mb: mb - 1,
        )
        below = _check_scheduler_limits(_spec(mem_per_node=mem, ntasks=1, cpus_per_task=1))
        assert any(MEM_EXCEEDS in w for w in below), mem
        monkeypatch.setattr(
            "forge.submit.slurm.get_job_memory_limit_mb",
            lambda mb=expected_mb: mb,
        )
        equal = _check_scheduler_limits(_spec(mem_per_node=mem, ntasks=1, cpus_per_task=1))
        assert not any(MEM_EXCEEDS in w for w in equal), mem


def test_no_mail_user_omits_mail_directives():
    text = _format_sbatch_directives(SlurmDirectives())
    assert "--mail-user" not in text
    assert "--mail-type" not in text
    script = generate_sbatch_script(_spec())
    assert "--mail-user" not in script
    assert "--mail-type" not in script


def test_explicit_mail_user_defaults_mail_type():
    text = _format_sbatch_directives(SlurmDirectives(mail_user="someone@example.com"))
    assert "--mail-user=someone@example.com" in text
    assert "--mail-type=BEGIN,END,FAIL" in text


def test_explicit_mail_user_and_mail_type_unchanged():
    text = _format_sbatch_directives(
        SlurmDirectives(mail_user="someone@example.com", mail_type="FAIL")
    )
    assert "--mail-user=someone@example.com" in text
    assert "--mail-type=FAIL" in text
    assert "--mail-type=BEGIN,END,FAIL" not in text
