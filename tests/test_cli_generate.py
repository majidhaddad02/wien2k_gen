"""CLI generate --gpu uses detected problem size, not hardcoded nmat/nkpt."""

from argparse import Namespace
from unittest.mock import MagicMock

from forge.cli_commands import generate as generate_cmd
from forge.core.topology import Topology
from forge.types import PipelineResult


class _DummyConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, *args, **kwargs) -> None:
        self.messages.append(" ".join(str(a) for a in args))

    def print_json(self, *args, **kwargs) -> None:
        return None


def _args(**overrides) -> Namespace:
    values = dict(
        max_cores=None,
        reserve_os_cores=None,
        mode=None,
        cores=None,
        omp=None,
        memory_limit=None,
        gpu=True,
        gpu_mixed_precision=False,
        dry_run=True,
        export=None,
        ignore_saturation=False,
        recalibrate=False,
        json_output=False,
        manual=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _patch_generate(monkeypatch, console, problem, gpu_rec_holder):
    monkeypatch.setattr("forge.cli_commands.generate.get_console", lambda: console)
    monkeypatch.setattr("forge.core.hardware.get_physical_cores", lambda: 8)
    monkeypatch.setattr(
        "forge.core.scheduler.detect",
        lambda max_cores=None: Topology(nodes=["n1"], cores_per_node=[8]),
    )
    monkeypatch.setattr(
        "forge.core.pipeline.run_pipeline",
        lambda **kw: PipelineResult(success=True, warnings=[]),
    )

    backend = MagicMock()
    backend.detect_problem_size.return_value = problem
    monkeypatch.setattr(
        "forge.backend_manager.get_current_backend", lambda: backend
    )
    monkeypatch.setattr(
        "forge.backends.gpu_backend.detect_gpu",
        lambda: [MagicMock(name="A100")],
    )

    def fake_rec(topo, nmat, nkpt, mode, mpi_ranks=None):
        gpu_rec_holder.append(
            {"nmat": nmat, "nkpt": nkpt, "mode": mode, "mpi_ranks": mpi_ranks}
        )
        return {"use_gpu": True, "gpu_count": 1, "nmat": nmat, "nkpt": nkpt}

    monkeypatch.setattr(
        "forge.backends.gpu_backend.get_gpu_recommendation", fake_rec
    )
    return backend


def test_gpu_uses_detected_nmat_nkpt(monkeypatch) -> None:
    console = _DummyConsole()
    captured: list[dict] = []
    _patch_generate(
        monkeypatch, console, {"nmat": 3200, "kpoints": 16}, captured
    )
    generate_cmd.handle(_args(), None)
    assert captured[0]["nmat"] == 3200
    assert captured[0]["nkpt"] == 16
    assert all("nmat is 0" not in m for m in console.messages)


def test_gpu_warns_when_nmat_is_zero(monkeypatch) -> None:
    console = _DummyConsole()
    captured: list[dict] = []
    _patch_generate(monkeypatch, console, {"nmat": 0, "kpoints": 0}, captured)
    generate_cmd.handle(_args(), None)
    assert captured[0]["nmat"] == 0
    assert captured[0]["nkpt"] == 0
    assert any("nmat is 0" in m for m in console.messages)


def test_gpu_mixed_precision_uses_detected_nmat(monkeypatch) -> None:
    console = _DummyConsole()
    captured: list[dict] = []
    mixed: list[int] = []
    _patch_generate(
        monkeypatch, console, {"nmat": 7200, "kpoints": 4}, captured
    )
    monkeypatch.setattr(
        "forge.backends.gpu_backend.get_mixed_precision_recommendation",
        lambda backend, nmat: mixed.append(nmat) or MagicMock(),
    )
    generate_cmd.handle(_args(gpu_mixed_precision=True), None)
    assert captured[0]["nmat"] == 7200
    assert mixed == [7200]


def test_extract_nmat_nkpt_from_object() -> None:
    class _P:
        nmat = 111
        kpoints = 9

    assert generate_cmd._extract_nmat_nkpt(_P()) == (111, 9)
    assert generate_cmd._extract_nmat_nkpt({"nmat": 50, "nkpt": 3}) == (50, 3)
