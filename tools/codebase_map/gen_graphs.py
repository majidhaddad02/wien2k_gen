"""DOT and Mermaid graph emitters."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .textutil import mermaid_id, q


def write_module_dependencies_dot(out_dir: Path, graphs: dict[str, Any]) -> None:
    edges = graphs["unique_production_edges"]
    packages: dict[str, list[str]] = defaultdict(list)
    for m in graphs["production_modules"]:
        pkg = ".".join(m.split(".")[:2]) if m.count(".") >= 1 else m
        packages[pkg].append(m)

    lines = [
        "digraph forge_imports {",
        '  rankdir=LR;',
        '  graph [label="FORGE production import graph (verified AST imports)", labelloc=t, fontsize=16];',
        '  node [shape=box, style=filled, fillcolor=white, fontname="Helvetica"];',
        '  edge [color="#444444"];',
        "",
    ]
    for i, (pkg, mods) in enumerate(sorted(packages.items())):
        lines.append(f"  subgraph cluster_{i} {{")
        lines.append(f"    label={q(pkg)};")
        lines.append("    color=gray;")
        for m in sorted(set(mods)):
            lines.append(f"    {q(m)};")
        lines.append("  }")
    for e in edges:
        lines.append(f'  {q(e["from"])} -> {q(e["to"])};')
    lines.append("}")
    (out_dir / "module_dependencies.dot").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_module_dependencies_mmd(out_dir: Path, graphs: dict[str, Any]) -> None:
    """Package-level mermaid view; full module graph is in DOT."""
    lines = [
        "flowchart LR",
        '  subgraph legend ["Legend"]',
        '    L1["Solid arrow: verified internal import"]',
        "  end",
        "",
    ]
    pkgs = sorted({e["from"] for e in graphs["package_deps"]} | {e["to"] for e in graphs["package_deps"]})
    for p in pkgs:
        lines.append(f"  {mermaid_id(p)}[{q(p)}]")
    for e in graphs["package_deps"]:
        lines.append(
            f"  {mermaid_id(e['from'])} -->|{e['count']}| {mermaid_id(e['to'])}"
        )
    (out_dir / "module_dependencies.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_architecture_overview_svg(out_dir: Path) -> None:
    """Minimal SVG overview when Graphviz/mermaid-cli are absent. Not a layout of the DOT graph."""
    boxes = [
        (40, 30, 200, 50, "Entry: cli / __main__ / wizard / cli_sbatch"),
        (40, 110, 200, 50, "CLI: argparse + get_handler"),
        (40, 190, 200, 50, "Pipeline: run_pipeline / build_auto"),
        (280, 110, 200, 50, "Advisor + hardware + scheduler"),
        (280, 190, 200, 50, "Backends: WIEN2k / QE / VASP / CP2K"),
        (520, 110, 200, 50, "Submit: slurm / pbs / lsf"),
        (520, 190, 200, 50, "External: run_lapw, sbatch, pw.x"),
    ]
    arrows = [
        ((140, 80), (140, 110)),
        ((140, 160), (140, 190)),
        ((240, 135), (280, 135)),
        ((140, 215), (280, 215)),
        ((480, 135), (520, 135)),
        ((480, 215), (520, 215)),
        ((620, 160), (620, 190)),
    ]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="280" viewBox="0 0 760 280">',
        "<title>FORGE high-level architecture (hand-laid SVG fallback)</title>",
        '<rect width="760" height="280" fill="white"/>',
        '<text x="20" y="18" font-size="12" font-family="sans-serif">'
        "Solid boxes/arrows: verified static relationships. See architecture_overview.mmd.</text>",
    ]
    for x, y, w, h, label in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#f7f7f7" stroke="#333"/>'
        )
        parts.append(
            f'<text x="{x + 10}" y="{y + 30}" font-size="11" font-family="sans-serif">{label}</text>'
        )
    for (x1, y1), (x2, y2) in arrows:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" marker-end="url(#arrow)"/>'
        )
    parts.insert(
        4,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#333"/></marker></defs>',
    )
    parts.append("</svg>")
    (out_dir / "architecture_overview.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_architecture_overview_mmd(out_dir: Path) -> None:
    lines = [
        "flowchart TB",
        '  subgraph entry ["Entry points - verified packaging"]',
        '    CLI["forge.cli:main"]',
        '    MAIN["python -m forge -> forge.__main__"]',
        '    SBATCH["forge.cli_sbatch:run_sbatch_cli"]',
        '    WIZ["forge.wizard:run_wizard"]',
        "  end",
        '  subgraph cli ["CLI dispatch - verified"]',
        '    PARSE["create_parser / argparse"]',
        '    REG["cli_commands.register_all"]',
        '    HAND["base.get_handler -> handle()"]',
        "  end",
        '  subgraph core ["Core orchestration - verified"]',
        '    PIPE["core.pipeline.run_pipeline"]',
        '    BUILD["core.builder.build_auto"]',
        '    TOPO["core.scheduler.detect"]',
        '    HW["core.hardware"]',
        '    WF["core.workflow / workflow_executor"]',
        "  end",
        '  subgraph be ["Backends - verified registry"]',
        '    BM["backend_manager shim"]',
        '    BR["backends._load_backends"]',
        '    W2K["Wien2kBackend"]',
        '    QE["QuantumEspressoBackend"]',
        '    VASP["VaspBackend"]',
        '    CP2K["CP2KBackend"]',
        "  end",
        '  subgraph opt ["Optimizer - verified"]',
        '    ADV["optimizer.advisor.suggest_optimal_resources"]',
        '    BAY["optimizer.bayesian"]',
        '    ML["ml.gnn_kpoint_predictor"]',
        "  end",
        '  subgraph sub ["Schedulers - verified"]',
        '    SL["submit.slurm"]',
        '    PBS["submit.pbs"]',
        '    LSF["submit.lsf"]',
        "  end",
        '  subgraph ext ["External programs - subprocess sites"]',
        '    DFT["WIEN2k run_lapw / QE pw.x / VASP / CP2K"]',
        '    SCH["sbatch squeue qsub bsub"]',
        "  end",
        "  CLI --> PARSE",
        "  MAIN --> CLI",
        "  PARSE --> REG",
        "  REG --> HAND",
        "  HAND --> PIPE",
        "  HAND --> WF",
        "  HAND --> ADV",
        "  PIPE --> TOPO",
        "  PIPE --> ADV",
        "  PIPE --> BUILD",
        "  PIPE -.-> BM",
        "  BUILD -.-> BM",
        "  BM --> BR",
        "  BR --> W2K",
        "  BR --> QE",
        "  BR --> VASP",
        "  BR --> CP2K",
        "  TOPO --> HW",
        "  HAND --> SL",
        "  HAND --> PBS",
        "  HAND --> LSF",
        "  SL --> SCH",
        "  PBS --> SCH",
        "  LSF --> SCH",
        "  W2K --> DFT",
        "  QE --> DFT",
        "  ADV --> ML",
        "  ADV --> BAY",
        "  SBATCH --> SL",
        "  WIZ --> BUILD",
        "  WIZ --> ADV",
        "",
        "  classDef dashed stroke-dasharray: 5 5",
    ]
    (out_dir / "architecture_overview.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cli_flow_mmd(out_dir: Path, cli_map: list[dict[str, Any]]) -> None:
    lines = [
        "flowchart TD",
        '  U["User: forge <command>"] --> EP["forge.cli.main"]',
        '  EP --> CFG["load_config / setup_logging"]',
        '  EP --> P["create_parser"]',
        '  P --> RA["cli_commands.register_all"]',
        '  EP --> GH["base.get_handler"]',
        '  GH --> DISP["handler(args, cfg)"]',
        "",
    ]
    for row in cli_map:
        cid = mermaid_id(row["command"])
        lines.append(f'  DISP --> {cid}[{q(row["command"] + " -> " + row["handler"])}]')
    lines.extend(
        [
            "",
            '  generate_forge_cli_commands_generate_handle --> PIPE["core.pipeline.run_pipeline"]',
            '  submit_forge_cli_commands_submit_handle --> SUB["submit.slurm / pbs / lsf"]',
            '  run_forge_cli_commands_run_handle --> WEX["workflow_executor.run_workflow_from_yaml"]',
            '  advise_forge_cli_commands_advise_handle --> ADV["optimizer.advisor / hardware"]',
        ]
    )
    # simpler explicit nodes
    extra = [
        "",
        '  subgraph representative ["Representative downstream - verified"]',
        '    G["generate.handle"] --> GP["run_pipeline"]',
        '    S["submit.handle"] --> SS["submit_slurm_job / SUBMIT_PROVIDERS"]',
        '    R["run.handle"] --> RW["run_workflow_from_yaml"]',
        '    A["advise.handle"] --> AA["CaseFileParser + detect topology"]',
        '    B["benchmark.handle"] --> BB["benchmark.real / synthetic"]',
        '    D["diagnose.handle"] --> DD["parse_scf_output"]',
        "  end",
        '  DISP --> G',
        '  DISP --> S',
        '  DISP --> R',
        '  DISP --> A',
        '  DISP --> B',
        '  DISP --> D',
    ]
    (out_dir / "cli_execution_flow.mmd").write_text("\n".join(lines[:8] + extra) + "\n", encoding="utf-8")


def write_execution_lifecycle_mmd(out_dir: Path) -> None:
    text = """sequenceDiagram
    title "General generate pipeline from core.pipeline.run_pipeline"
    participant User
    participant CLI as cli_commands.generate.handle
    participant Sched as core.scheduler.detect
    participant Pipe as core.pipeline.run_pipeline
    participant Backend as get_current_backend
    participant Adv as optimizer.advisor
    participant Build as core.builder.build_auto
    participant Disk as .machines / config file

    User->>CLI: forge generate
    CLI->>Sched: detect(max_cores)
    Sched-->>CLI: Topology
    CLI->>Pipe: run_pipeline(topo, user_suggestion)
    Pipe->>Backend: detect_problem_size()
    Backend-->>Pipe: ProblemSize
    Pipe->>Adv: suggest_optimal_resources(topo)
    Adv-->>Pipe: ResourceSuggestion
    Pipe->>Pipe: preflight_check
    alt dry_run
        Pipe->>Backend: generate_input(topo, suggestion)
        Backend-->>Pipe: config string
    else write
        Pipe->>Build: build_auto(topo, suggestion)
        Build->>Backend: generate_input
        Build->>Disk: atomic_write
        Pipe->>Pipe: validate_machines or backend.validate_config
    end
    Pipe-->>CLI: PipelineResult
    CLI-->>User: path / preview / errors
"""
    (out_dir / "execution_lifecycle.mmd").write_text(text, encoding="utf-8")

    wien = """sequenceDiagram
    title "WIEN2k generate and submit verified static path"
    participant User
    participant Gen as generate.handle
    participant Pipe as run_pipeline
    participant W2K as Wien2kBackend
    participant Sub as submit.handle
    participant SL as submit.slurm.submit_slurm_job
    participant OS as sbatch subprocess

    User->>Gen: forge generate
    Gen->>Pipe: run_pipeline
    Pipe->>W2K: detect_problem_size / generate_input
    Note over W2K: writes .machines via builder
    User->>Sub: forge submit
    Sub->>Sub: resolve_scheduler / parse .machines
    alt scheduler slurm
        Sub->>SL: submit_slurm_job(spec)
        SL->>OS: subprocess.run sbatch
    else scheduler pbs or lsf
        Sub->>Sub: SUBMIT_PROVIDERS[scheduler].submit
    end
"""
    (out_dir / "wien2k_execution.mmd").write_text(wien, encoding="utf-8")

    qe = """sequenceDiagram
    title "Quantum ESPRESSO path verified modules selection is dynamic"
    participant User
    participant CLI
    participant BM as backends.auto_detect / set_backend
    participant QE as QuantumEspressoBackend
    participant Exec as quantum_espresso.executor
    participant OS as subprocess.Popen

    User->>CLI: forge --backend qe generate
    CLI->>BM: BackendCode.QUANTUM_ESPRESSO
    BM->>QE: get_backend
    QE->>QE: detect_problem_size from .in / .pw.in
    QE->>QE: generate_input / run_qe_optimized.sh
    Note over Exec: executor.py contains subprocess.Popen
    Exec->>OS: launch pw.x or related binary
"""
    (out_dir / "qe_execution.mmd").write_text(qe, encoding="utf-8")

    sched = """sequenceDiagram
    title "Scheduler-mediated submit from cli_commands.submit.handle"
    participant User
    participant Sub as submit.handle
    participant Det as resolve_scheduler / detect
    participant SL as submit.slurm
    participant PBS as submit.pbs.PBSSubmitProvider
    participant LSF as submit.lsf.LSFSubmitProvider

    User->>Sub: forge submit --scheduler auto
    Sub->>Det: resolve_scheduler
    alt slurm
        Sub->>SL: submit_slurm_job
    else pbs
        Sub->>PBS: provider.submit
    else lsf
        Sub->>LSF: provider.submit
    else unknown
        Sub-->>User: success false
    end
"""
    (out_dir / "scheduler_execution.mmd").write_text(sched, encoding="utf-8")


def write_test_ci_mmd(out_dir: Path) -> None:
    test_arch = """flowchart TB
  subgraph unit ["Unit tests under tests/"]
    T1["test_advisor.py"]
    T2["test_builder.py"]
    T3["test_scheduler.py"]
    T4["test_wien2k_backend.py"]
    T5["test_hardware.py"]
    T6["test_bayesian*.py"]
    T7["test_gnn_kpoint_predictor.py"]
  end
  subgraph integ ["Integration / hardware markers"]
    I1["test_integration.py"]
    I2["integration_test.py"]
    I3["test_robustness.py"]
  end
  subgraph rf ["ReFrame"]
    R1["tests/reframe/wien2k_gen_test.py"]
    R2["tests/reframe/reframe_config.py"]
  end
  subgraph fx ["Fixtures"]
    F1["tests/fixtures/*.struct *.scf *.json"]
    F2["tests/conftest.py"]
  end
  F2 --> unit
  F1 --> unit
  F2 --> integ
"""
    (out_dir / "test_architecture.mmd").write_text(test_arch, encoding="utf-8")

    ci = """flowchart TD
  subgraph ci_yml [".github/workflows/ci.yml"]
    A["push/PR to master"] --> B["matrix Python 3.9-3.12"]
    B --> C["pip install -e .[dev,hpc]"]
    C --> D["ruff check src/ tests/"]
    C --> E["mypy src/"]
    C --> F["pytest --cov=forge --cov-fail-under=15"]
    C --> G["python -c import forge"]
    F --> H["upload coverage.xml"]
  end
  subgraph rf_yml [".github/workflows/reframe_benchmark.yml"]
    P["push/PR master main develop"] --> Q["matrix Python 3.10-3.12"]
    Q --> R["pip install -e .[dev] and reframe-hpc"]
    R --> S["reframe -C tests/reframe --tag smoke"]
    R --> T["reframe --tag benchmark"]
    S --> U["upload logs"]
    T --> U
  end
"""
    (out_dir / "ci_pipeline.mmd").write_text(ci, encoding="utf-8")


def write_callgraph_focus(out_dir: Path, analysis: dict[str, Any]) -> None:
    """Focused verified call edges for pipeline / CLI / backends."""
    interesting_callers = (
        "forge.cli:",
        "forge.cli_commands.generate:",
        "forge.cli_commands.submit:",
        "forge.core.pipeline:",
        "forge.core.builder:",
        "forge.backends:",
        "forge.optimizer.advisor:",
    )
    lines = [
        "flowchart LR",
        '  subgraph note ["Verified Name-resolved calls only"]',
        '    N["Dashed targets omitted; see call_graph.json"]',
        "  end",
    ]
    per_prefix: dict[str, list[tuple[str, str]]] = {p: [] for p in interesting_callers}
    seen: set[tuple[str, str]] = set()
    for m in analysis["modules"]:
        if m["category"] != "production":
            continue
        for c in m["calls"]:
            if c["confidence"] != "verified" or not c.get("resolved"):
                continue
            key = (c["caller"], c["resolved"])
            if key in seen:
                continue
            for p in interesting_callers:
                if c["caller"].startswith(p):
                    seen.add(key)
                    per_prefix[p].append(key)
                    break
    for p in interesting_callers:
        for caller, resolved in per_prefix[p][:12]:
            a = mermaid_id(caller)
            b = mermaid_id(resolved)
            lines.append(f"  {a}[{q(caller)}] --> {b}[{q(resolved)}]")
    lines.append("")
    lines.append("%% Sample of verified calls; full list is in call_graph.json")
    (out_dir / "call_graph_focused.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
