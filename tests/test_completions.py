"""Shell completion scripts must complete CLI subcommands without bash-completion."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASH_FORGE = REPO / "completions" / "forge.bash"
BASH_SBATCH = REPO / "completions" / "forge_sbatch.bash"


def _complete(script: Path, words: list[str], func: str) -> list[str]:
    quoted = " ".join(f"'{w}'" for w in words)
    cmd = f"""
source '{script}'
COMP_WORDS=({quoted})
COMP_CWORD=$(( ${{#COMP_WORDS[@]}} - 1 ))
COMPREPLY=()
{func}
printf '%s\\n' "${{COMPREPLY[@]}}"
"""
    out = subprocess.check_output(["bash", "--noprofile", "--norc", "-c", cmd], text=True)
    return [line for line in out.splitlines() if line]


def test_forge_completes_subcommands() -> None:
    reply = _complete(BASH_FORGE, ["forge", ""], "_forge")
    for cmd in ("generate", "submit", "hardware", "analyze-bands", "history", "calibrate"):
        assert cmd in reply


def test_forge_prefix_generate() -> None:
    reply = _complete(BASH_FORGE, ["forge", "gen"], "_forge")
    assert reply == ["generate"]


def test_forge_generate_mode_values() -> None:
    reply = _complete(BASH_FORGE, ["forge", "generate", "--mode", ""], "_forge")
    assert set(reply) == {"mpi", "hybrid", "kpoint"}


def test_forge_generate_recalibrate_flag() -> None:
    reply = _complete(BASH_FORGE, ["forge", "generate", "--"], "_forge")
    assert "--recalibrate" in reply
    assert "--ignore-saturation" in reply


def test_forge_skips_global_flags_to_find_command() -> None:
    reply = _complete(BASH_FORGE, ["forge", "-v", "--json", "gen"], "_forge")
    assert reply == ["generate"]


def test_forge_history_extra_flags() -> None:
    reply = _complete(BASH_FORGE, ["forge", "history", "--"], "_forge")
    assert "--export" in reply
    assert "--format" in reply


def test_sbatch_completes_actions() -> None:
    reply = _complete(BASH_SBATCH, ["forge_sbatch", ""], "_forge_sbatch")
    for cmd in ("generate", "validate", "preview", "submit"):
        assert cmd in reply
