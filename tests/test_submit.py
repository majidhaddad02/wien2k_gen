"""Tests for forge submit resource resolution from .machines allocation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from forge.cli_commands import submit as submit_cmd
from forge.core.topology import Topology
from forge.types import PipelineResult, ResourceSuggestion


HYBRID_MACHINES = """\
lapw0: node01: 1
1: node01: 13
omp_global: 8
"""

MISSING_MSG = submit_cmd.MISSING_MACHINES_MSG


class _DummyConsole:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.printed: list[object] = []
        self.inputs: list[str] = []
        self._answers = list(answers or [])

    def print(self, *args, **kwargs) -> None:
        self.printed.append(args[0] if args else "")

    def input(self, prompt: str = "") -> str:
        self.inputs.append(prompt)
        if self._answers:
            return self._answers.pop(0)
        return "n"


def _topo(total: int = 128) -> Topology:
    return Topology(nodes=["node01"], cores_per_node=[total])


def _args(**kwargs) -> Namespace:
    defaults = dict(
        scheduler="slurm",
        partition="",
        nodes=1,
        ntasks=0,
        cpus_per_task=0,
        time="24:00:00",
        mem="16G",
        job_name="wien2k_job",
        dependency="",
        dry_run=True,
        export=None,
        auto_generate=False,
        no_auto_generate=False,
        json_output=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def _write_machines(path: Path, content: str = HYBRID_MACHINES) -> Path:
    machines = path / ".machines"
    machines.write_text(content)
    return machines


def test_hybrid_machines_uses_clamped_allocation_not_topology(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_machines(tmp_path)
    topo = _topo(128)
    console = _DummyConsole()
    resolved = submit_cmd.resolve_submit_resources(_args(), topo, console)
    assert resolved["success"] is True
    assert resolved["ntasks"] == 13
    assert resolved["cpus_per_task"] == 8
    assert resolved["ntasks"] != topo.total_cores
    assert resolved["cpus_per_task"] != 1


def test_cli_ntasks_and_cpus_per_task_override_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_machines(tmp_path)
    resolved = submit_cmd.resolve_submit_resources(
        _args(ntasks=4, cpus_per_task=2),
        _topo(128),
        _DummyConsole(),
    )
    assert resolved["success"] is True
    assert resolved["ntasks"] == 4
    assert resolved["cpus_per_task"] == 2


def test_missing_machines_json_fails_without_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    console = _DummyConsole(answers=["y"])
    monkeypatch.setattr(submit_cmd.sys.stdin, "isatty", lambda: True)
    resolved = submit_cmd.resolve_submit_resources(
        _args(json_output=True),
        _topo(),
        console,
    )
    assert resolved["success"] is False
    assert MISSING_MSG in resolved["errors"]
    assert console.inputs == []


def test_missing_machines_no_auto_generate_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit_cmd.sys.stdin, "isatty", lambda: True)
    resolved = submit_cmd.resolve_submit_resources(
        _args(no_auto_generate=True),
        _topo(),
        _DummyConsole(answers=["y"]),
    )
    assert resolved["success"] is False
    assert MISSING_MSG in resolved["errors"]


def test_missing_machines_non_tty_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit_cmd.sys.stdin, "isatty", lambda: False)
    resolved = submit_cmd.resolve_submit_resources(_args(), _topo(), _DummyConsole())
    assert resolved["success"] is False
    assert MISSING_MSG in resolved["errors"]


def test_missing_machines_auto_generate_uses_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    called = {}

    def fake_pipeline(topo, **kwargs):
        called["topo"] = topo
        return PipelineResult(
            success=True,
            suggestion=ResourceSuggestion(
                recommended_total_cores=13,
                omp_threads_per_rank=8,
            ),
        )

    monkeypatch.setattr("forge.core.pipeline.run_pipeline", fake_pipeline)
    resolved = submit_cmd.resolve_submit_resources(
        _args(auto_generate=True),
        _topo(128),
        _DummyConsole(),
    )
    assert called
    assert resolved["success"] is True
    assert resolved["ntasks"] == 13
    assert resolved["cpus_per_task"] == 8


def test_missing_machines_tty_prompt_yes_generates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit_cmd.sys.stdin, "isatty", lambda: True)
    console = _DummyConsole(answers=["y"])
    monkeypatch.setattr(
        "forge.core.pipeline.run_pipeline",
        lambda topo, **kw: PipelineResult(
            success=True,
            suggestion=ResourceSuggestion(
                recommended_total_cores=13,
                omp_threads_per_rank=8,
            ),
        ),
    )
    resolved = submit_cmd.resolve_submit_resources(_args(), _topo(128), console)
    assert console.inputs
    assert resolved["success"] is True
    assert resolved["ntasks"] == 13
    assert resolved["cpus_per_task"] == 8


def test_missing_machines_tty_prompt_no_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit_cmd.sys.stdin, "isatty", lambda: True)
    console = _DummyConsole(answers=["n"])
    resolved = submit_cmd.resolve_submit_resources(_args(), _topo(), console)
    assert console.inputs
    assert resolved["success"] is False
    assert MISSING_MSG in resolved["errors"]


def test_handle_slurm_passes_machines_allocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_machines(tmp_path)
    captured = {}

    def fake_submit(spec, dry_run=False, script_path=None):
        captured["ntasks"] = spec.directives.ntasks
        captured["cpus_per_task"] = spec.directives.cpus_per_task
        captured["exec_command"] = spec.exec_command
        return {"success": True, "job_id": "1", "script_path": tmp_path / "job.sh"}

    monkeypatch.setattr(submit_cmd, "get_console", lambda: _DummyConsole())
    monkeypatch.setattr(
        "forge.core.scheduler.detect",
        lambda max_cores=None: _topo(128),
    )
    monkeypatch.setattr("forge.submit.slurm.submit_slurm_job", fake_submit)
    monkeypatch.setattr(submit_cmd, "get_exec_command", lambda: "run_lapw -p -so")

    result = submit_cmd.handle(_args(json_output=True, dry_run=True), None)
    assert result["success"] is True
    assert result["ntasks"] == 13
    assert result["cpus_per_task"] == 8
    assert captured["ntasks"] == 13
    assert captured["cpus_per_task"] == 8
    assert captured["exec_command"] == "run_lapw -p -so"


def test_handle_pbs_uses_get_exec_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_machines(tmp_path)
    captured = {}

    class FakeProvider:
        def submit(self, topo, exec_command, directives=None, script_path=None, dry_run=False, **kwargs):
            captured["exec_command"] = exec_command
            return {"success": True, "job_id": "pbs.1", "script_path": tmp_path / "pbs.sh"}

    monkeypatch.setattr(submit_cmd, "get_console", lambda: _DummyConsole())
    monkeypatch.setattr("forge.core.scheduler.detect", lambda max_cores=None: _topo(128))
    monkeypatch.setattr(submit_cmd, "get_exec_command", lambda: "run_lapw -p -so")
    monkeypatch.setattr(
        "forge.submit.SUBMIT_PROVIDERS",
        {"pbs": FakeProvider, "lsf": FakeProvider},
    )

    result = submit_cmd.handle(_args(scheduler="pbs", json_output=True), None)
    assert result["success"] is True
    assert captured["exec_command"] == "run_lapw -p -so"
    assert result["ntasks"] == 13
    assert result["cpus_per_task"] == 8


def test_handle_lsf_uses_get_exec_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_machines(tmp_path)
    captured = {}

    class FakeProvider:
        def submit(self, topo, exec_command, directives=None, script_path=None, dry_run=False, **kwargs):
            captured["exec_command"] = exec_command
            return {"success": True, "job_id": "lsf.1", "script_path": tmp_path / "lsf.sh"}

    monkeypatch.setattr(submit_cmd, "get_console", lambda: _DummyConsole())
    monkeypatch.setattr("forge.core.scheduler.detect", lambda max_cores=None: _topo(128))
    monkeypatch.setattr(submit_cmd, "get_exec_command", lambda: "runsp_lapw -p")
    monkeypatch.setattr(
        "forge.submit.SUBMIT_PROVIDERS",
        {"pbs": FakeProvider, "lsf": FakeProvider},
    )

    result = submit_cmd.handle(_args(scheduler="lsf", json_output=True), None)
    assert result["success"] is True
    assert captured["exec_command"] == "runsp_lapw -p"
    assert result["ntasks"] == 13
    assert result["cpus_per_task"] == 8


def test_handle_missing_machines_json_populates_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit_cmd, "get_console", lambda: _DummyConsole())
    monkeypatch.setattr("forge.core.scheduler.detect", lambda max_cores=None: _topo(128))
    monkeypatch.setattr(submit_cmd.sys.stdin, "isatty", lambda: False)
    result = submit_cmd.handle(_args(json_output=True), None)
    assert result["success"] is False
    assert result.get("errors")
    assert MISSING_MSG in result["errors"]


def test_register_adds_cpus_per_task_and_auto_generate():
    parser_ns = SimpleNamespace()
    added = {}

    class FakeParser:
        def add_argument(self, *flags, **kwargs):
            for flag in flags:
                added[flag] = kwargs
            return MagicMock()

        def add_mutually_exclusive_group(self):
            return self

    class FakeSub:
        def add_parser(self, name, **kwargs):
            parser_ns.name = name
            return FakeParser()

    submit_cmd.register(FakeSub())
    assert parser_ns.name == "submit"
    assert "--cpus-per-task" in added
    assert "--auto-generate" in added
    assert "--no-auto-generate" in added
    assert "--ntasks" in added
