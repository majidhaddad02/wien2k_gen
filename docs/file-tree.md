# Complete Repository File Diagram

This page lists **every tracked source file** in the FORGE / `wien2k_gen` tree. Generated caches (`__pycache__/`, `.pytest_cache/`, `*.pyc`, `src/forge.egg-info/`, `coverage.xml`) are named as directories only.

Tab-complete CLI flags from `completions/forge.bash` and `completions/forge.zsh`.

---

## 1. Top-level layout

```mermaid
flowchart TB
    ROOT["wien2k_gen repository"]
    ROOT --> SRC["src/forge/"]
    ROOT --> TESTS["tests/"]
    ROOT --> DOCS["docs/"]
    ROOT --> EX["examples/"]
    ROOT --> COMP["completions/"]
    ROOT --> OFF["offline_packages/"]
    ROOT --> GH[".github/workflows/"]
    ROOT --> CFG["root config and install"]
    SRC --> CLI["cli.py plus cli_commands/"]
    SRC --> BE["backends/"]
    SRC --> CORE["core/"]
    SRC --> OPT["optimizer/"]
    SRC --> ML["ml/"]
    SRC --> SUB["submit/"]
    SRC --> UT["utils/"]
    SRC --> UI["ui/ plus wizard"]
    CLI --> BE
    CORE --> BE
    OPT --> CORE
    SUB --> UT
```

---

## 2. Root files

```mermaid
flowchart LR
    subgraph rootfiles["root"]
        A1["README.md"]
        A2["LICENSE.md"]
        A3["CITATION.cff"]
        A4["pyproject.toml"]
        A5["Makefile"]
        A6["install.sh"]
        A7["environment.yml"]
        A8["download_and_train.py"]
        A9["Dockerfile"]
        A10["docker-compose.yml"]
        A11[".dockerignore"]
        A12["Singularity.def"]
        A13[".gitignore"]
        A14[".pre-commit-config.yaml"]
        A15["parallel_options"]
    end
```

| File | Role |
|------|------|
| `README.md` | Project overview |
| `LICENSE.md` | MIT license |
| `CITATION.cff` | Citation metadata |
| `pyproject.toml` | Package, extras, entry points |
| `Makefile` | install, test, docker, completions |
| `install.sh` | Production installer |
| `environment.yml` | Conda environment |
| `download_and_train.py` | ML dataset / train helper |
| `Dockerfile` | multi-stage `runtime` / `dev` |
| `docker-compose.yml` | forge / dev / test services |
| `.dockerignore` | Docker build context filter |
| `Singularity.def` | Apptainer image |
| `.gitignore` | Git ignore rules |
| `.pre-commit-config.yaml` | pre-commit hooks |
| `parallel_options` | sample / generated WIEN2k options |

---

## 3. GitHub workflows

```mermaid
flowchart LR
    subgraph gh[".github/workflows/"]
        G1["ci.yml"]
        G2["reframe_benchmark.yml"]
    end
```

---

## 4. Completions

```mermaid
flowchart LR
    subgraph comp["completions/"]
        C1["forge.bash"]
        C2["forge.zsh"]
        C3["forge_sbatch.bash"]
        C4["forge_sbatch.zsh"]
        C5["forge_wizard.bash"]
        C6["forge_wizard.zsh"]
    end
```

---

## 5. Documentation (`docs/`)

```mermaid
flowchart TB
    subgraph docs["docs/"]
        D1["index.md"]
        D2["installation.md"]
        D3["user-guide.md"]
        D4["workflow.md"]
        D5["machines-guide.md"]
        D6["job-submission.md"]
        D7["parallel-modes.md"]
        D8["preprocessing-convergence.md"]
        D9["troubleshooting.md"]
        D10["examples.md"]
        D11["contributing.md"]
        D12["ml_dataset.md"]
        D13["api-reference.md"]
        D14["file-tree.md"]
        D15["conf.py"]
        D16["index.rst"]
        D17["api.rst"]
    end
    D1 --> D2
    D1 --> D3
    D1 --> D13
    D1 --> D14
```

| File | Role |
|------|------|
| `index.md` | Docs home |
| `installation.md` | `install.sh`, source, Docker |
| `user-guide.md` | Every CLI flag with examples |
| `workflow.md` | `init_lapw` to submit |
| `machines-guide.md` | `.machines` generate / read |
| `job-submission.md` | four submit models |
| `parallel-modes.md` | kpoint / hybrid / mpi / fine_grain |
| `preprocessing-convergence.md` | RKmax, k-mesh, mixing |
| `troubleshooting.md` | errors and diagnostics |
| `examples.md` | scenario commands |
| `contributing.md` | development |
| `ml_dataset.md` | GNN / Bayesian / history |
| `api-reference.md` | Python API (do not casually edit) |
| `file-tree.md` | this diagram |
| `conf.py` | leftover Sphinx config |
| `index.rst` | stub pointing at Markdown |
| `api.rst` | stub pointing at `api-reference.md` |

---

## 6. Examples

```mermaid
flowchart TB
    subgraph examples["examples/"]
        E0["README.md"]
        subgraph si["01_si_semiconductor/"]
            S1["Si.struct"]
            S2["Si.in1"]
            S3["Si.in2"]
            S4["wien2k_gen.yaml"]
        end
        subgraph cu["02_cu_metal/"]
            U1["Cu.struct"]
            U2["Cu.in1"]
            U3["Cu.in2"]
            U4["wien2k_gen.yaml"]
        end
        subgraph fe["03_fe_magnetic/"]
            F1["Fe.struct"]
            F2["Fe.in1"]
            F3["Fe.in2"]
            F4["Fe.inst"]
            F5["wien2k_gen.yaml"]
        end
        E0 --> si
        E0 --> cu
        E0 --> fe
    end
```

---

## 7. Offline wheels

```mermaid
flowchart TB
    subgraph off["offline_packages/"]
        O0["requirements-offline.txt"]
        subgraph wheels["packaging_offline/"]
            W1["PyYAML-6.0.1-cp39-...whl"]
            W2["linkify_it_py-2.0.3-py3-none-any.whl"]
            W3["markdown_it_py-3.0.0-py3-none-any.whl"]
            W4["mdit_py_plugins-0.4.2-py3-none-any.whl"]
            W5["mdurl-0.1.2-py3-none-any.whl"]
            W6["numpy-1.24.4-cp39-...whl"]
            W7["platformdirs-4.3.6-py3-none-any.whl"]
            W8["psutil-5.9.8-cp36-abi3-...whl"]
            W9["pygments-2.18.0-py3-none-any.whl"]
            W10["rich-13.7.1-py3-none-any.whl"]
            W11["textual-0.85.2-py3-none-any.whl"]
            W12["typing_extensions-4.12.2-py3-none-any.whl"]
            W13["uc_micro_py-1.0.3-py3-none-any.whl"]
        end
        O0 --> wheels
    end
```

Exact wheel filenames:

- `offline_packages/requirements-offline.txt`
- `offline_packages/packaging_offline/PyYAML-6.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- `offline_packages/packaging_offline/linkify_it_py-2.0.3-py3-none-any.whl`
- `offline_packages/packaging_offline/markdown_it_py-3.0.0-py3-none-any.whl`
- `offline_packages/packaging_offline/mdit_py_plugins-0.4.2-py3-none-any.whl`
- `offline_packages/packaging_offline/mdurl-0.1.2-py3-none-any.whl`
- `offline_packages/packaging_offline/numpy-1.24.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- `offline_packages/packaging_offline/platformdirs-4.3.6-py3-none-any.whl`
- `offline_packages/packaging_offline/psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- `offline_packages/packaging_offline/pygments-2.18.0-py3-none-any.whl`
- `offline_packages/packaging_offline/rich-13.7.1-py3-none-any.whl`
- `offline_packages/packaging_offline/textual-0.85.2-py3-none-any.whl`
- `offline_packages/packaging_offline/typing_extensions-4.12.2-py3-none-any.whl`
- `offline_packages/packaging_offline/uc_micro_py-1.0.3-py3-none-any.whl`

---

## 8. Package entry and CLI

```mermaid
flowchart TB
    subgraph pkg["src/forge/"]
        P0["__init__.py"]
        P1["__main__.py"]
        P2["cli.py"]
        P3["cli_sbatch.py"]
        P4["wizard.py"]
        P5["wizard_sbatch.py"]
        P6["config.py"]
        P7["types.py"]
        P8["exceptions.py"]
        P9["logging_config.py"]
        P10["backend_manager.py"]
    end
    P2 --> CMD["cli_commands/"]
    P3 --> SUB["submit/"]
    P10 --> BE["backends/"]
```

### `src/forge/cli_commands/`

```mermaid
flowchart LR
    subgraph cmds["cli_commands/"]
        Q0["__init__.py"]
        Q1["base.py"]
        Q2["_utils.py"]
        Q3["generate.py"]
        Q4["submit.py"]
        Q5["benchmark.py"]
        Q6["diagnostics.py"]
        Q7["hardware.py"]
        Q8["analyze.py"]
        Q9["tui.py"]
        Q10["monitor.py"]
        Q11["run.py"]
        Q12["workflow.py"]
        Q13["diagnose.py"]
        Q14["optimize.py"]
        Q15["screen.py"]
        Q16["predict.py"]
        Q17["advise.py"]
        Q18["converge.py"]
        Q19["history.py"]
        Q20["analyze_bands.py"]
        Q21["calibrate.py"]
    end
```

---

## 9. Backends

```mermaid
flowchart TB
    subgraph backends["src/forge/backends/"]
        B0["__init__.py"]
        B1["base.py"]
        B2["elpa_selector.py"]
        B3["gpu_backend.py"]
        B4["vasp.py"]
        B5["cp2k.py"]
        subgraph w2k["wien2k/"]
            W0["__init__.py"]
            W1["core.py"]
            W2["parsers.py"]
        end
        subgraph qe["quantum_espresso/"]
            QE0["__init__.py"]
            QE1["backend.py"]
            QE2["config_generator.py"]
            QE3["executor.py"]
            QE4["parser.py"]
        end
        B1 --> w2k
        B1 --> qe
        B1 --> B4
        B1 --> B5
    end
```

---

## 10. Core

```mermaid
flowchart TB
    subgraph core["src/forge/core/"]
        C0["__init__.py"]
        C1["pipeline.py"]
        C2["builder.py"]
        C3["scheduler.py"]
        C4["topology.py"]
        C5["case_parser.py"]
        C6["locator.py"]
        C7["constants.py"]
        C8["energy.py"]
        C9["perf_counters.py"]
        C10["electronic_structure.py"]
        C11["materials_project.py"]
        C12["terminal_monitor.py"]
        C13["workflow.py"]
        C14["workflow_executor.py"]
        subgraph hw["hardware/"]
            H0["__init__.py"]
            H1["cpu.py"]
            H2["detection.py"]
            H3["system.py"]
            H4["types.py"]
            H5["wrapper.py"]
        end
        C1 --> hw
        C1 --> C3
        C1 --> C2
    end
```

---

## 11. Optimizer, ML, submit, utils, UI, benchmark

```mermaid
flowchart TB
    subgraph opt["src/forge/optimizer/"]
        O0["__init__.py"]
        O1["advisor.py"]
        O2["parallel.py"]
        O3["convergence.py"]
        O4["history.py"]
        O5["profiler.py"]
        O6["bayesian_tuner.py"]
        O7["ml_predict.py"]
        O8["gpu_detector.py"]
        subgraph bayes["bayesian/"]
            Y0["__init__.py"]
            Y1["core.py"]
            Y2["gp.py"]
            Y3["kernels.py"]
            Y4["acquisition.py"]
            Y5["elements.py"]
            Y6["sampling.py"]
            Y7["constraints.py"]
            Y8["bohb.py"]
            Y9["dpp.py"]
        end
        subgraph mon["monitor/"]
            M0["__init__.py"]
            M1["convergence.py"]
            M2["checkpoint.py"]
            M3["engine.py"]
            M4["types.py"]
        end
        O6 --> bayes
        O3 --> mon
    end
```

```mermaid
flowchart LR
    subgraph ml["src/forge/ml/"]
        ML0["__init__.py"]
        ML1["gnn_kpoint_predictor.py"]
        ML2["data_pipeline.py"]
        ML3["gnn_kpoint_v1.npz"]
    end
    subgraph sub["src/forge/submit/"]
        SU0["__init__.py"]
        SU1["slurm.py"]
        SU2["pbs.py"]
        SU3["lsf.py"]
    end
    subgraph ut["src/forge/utils/"]
        U0["__init__.py"]
        U1["validation.py"]
        U2["parallel_options.py"]
        U3["diagnostic.py"]
        U4["export.py"]
        U5["scratch.py"]
        U6["atomic_write.py"]
        U7["filelock.py"]
        U8["subprocess_utils.py"]
    end
    subgraph ui["src/forge/ui/"]
        I0["__init__.py"]
        I1["rich_ui.py"]
        I2["analysis.py"]
    end
    subgraph bench["src/forge/benchmark/"]
        N0["__init__.py"]
        N1["synthetic.py"]
        N2["real.py"]
        N3["report.py"]
    end
```

---

## 12. Tests

```mermaid
flowchart TB
    subgraph tests["tests/"]
        T0["__init__.py"]
        T1["conftest.py"]
        T2["integration_test.py"]
        subgraph fix["fixtures/"]
            X1["case_minimal.struct"]
            X2["scf_converged.scf"]
            X3["scf_not_converged.scf"]
            X4["hardware_avx512_ib.json"]
            X5["topology_slurm.json"]
        end
        subgraph rf["reframe/"]
            R1["reframe_config.py"]
            R2["wien2k_gen_test.py"]
        end
        T3["test_advisor.py"]
        T4["test_analysis.py"]
        T5["test_auto_detect.py"]
        T6["test_bayesian.py"]
        T7["test_bayesian_core.py"]
        T8["test_bohb_dpp.py"]
        T9["test_builder.py"]
        T10["test_case_parser.py"]
        T11["test_completions.py"]
        T12["test_config.py"]
        T13["test_convergence_opt.py"]
        T14["test_diagnose.py"]
        T15["test_elpa_selector.py"]
        T16["test_gnn_kpoint_predictor.py"]
        T17["test_hardware.py"]
        T18["test_integration.py"]
        T19["test_new_features.py"]
        T20["test_parallel.py"]
        T21["test_parallel_options.py"]
        T22["test_perf_counters.py"]
        T23["test_robustness.py"]
        T24["test_scheduler.py"]
        T25["test_slurm.py"]
        T26["test_submit.py"]
        T27["test_topology.py"]
        T28["test_types.py"]
        T29["test_utils.py"]
        T30["test_wien2k_backend.py"]
        T31["test_wien2k_flags.py"]
        T32["test_wizard.py"]
        T1 --> fix
        T1 --> rf
    end
```

---

## 13. Full path inventory

Omitted as files: `__pycache__/**`, `.pytest_cache/**`, `*.pyc`, `src/forge.egg-info/**`, `.monkeycode-tmp-files/**`, `coverage.xml`.

```
.gitignore
.pre-commit-config.yaml
.dockerignore
.github/workflows/ci.yml
.github/workflows/reframe_benchmark.yml
CITATION.cff
Dockerfile
LICENSE.md
Makefile
README.md
Singularity.def
completions/forge.bash
completions/forge.zsh
completions/forge_sbatch.bash
completions/forge_sbatch.zsh
completions/forge_wizard.bash
completions/forge_wizard.zsh
docker-compose.yml
docs/api-reference.md
docs/api.rst
docs/conf.py
docs/contributing.md
docs/examples.md
docs/file-tree.md
docs/index.md
docs/index.rst
docs/installation.md
docs/job-submission.md
docs/machines-guide.md
docs/ml_dataset.md
docs/parallel-modes.md
docs/preprocessing-convergence.md
docs/troubleshooting.md
docs/user-guide.md
docs/workflow.md
download_and_train.py
environment.yml
examples/README.md
examples/01_si_semiconductor/Si.in1
examples/01_si_semiconductor/Si.in2
examples/01_si_semiconductor/Si.struct
examples/01_si_semiconductor/wien2k_gen.yaml
examples/02_cu_metal/Cu.in1
examples/02_cu_metal/Cu.in2
examples/02_cu_metal/Cu.struct
examples/02_cu_metal/wien2k_gen.yaml
examples/03_fe_magnetic/Fe.in1
examples/03_fe_magnetic/Fe.in2
examples/03_fe_magnetic/Fe.inst
examples/03_fe_magnetic/Fe.struct
examples/03_fe_magnetic/wien2k_gen.yaml
install.sh
offline_packages/requirements-offline.txt
offline_packages/packaging_offline/PyYAML-6.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
offline_packages/packaging_offline/linkify_it_py-2.0.3-py3-none-any.whl
offline_packages/packaging_offline/markdown_it_py-3.0.0-py3-none-any.whl
offline_packages/packaging_offline/mdit_py_plugins-0.4.2-py3-none-any.whl
offline_packages/packaging_offline/mdurl-0.1.2-py3-none-any.whl
offline_packages/packaging_offline/numpy-1.24.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
offline_packages/packaging_offline/platformdirs-4.3.6-py3-none-any.whl
offline_packages/packaging_offline/psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
offline_packages/packaging_offline/pygments-2.18.0-py3-none-any.whl
offline_packages/packaging_offline/rich-13.7.1-py3-none-any.whl
offline_packages/packaging_offline/textual-0.85.2-py3-none-any.whl
offline_packages/packaging_offline/typing_extensions-4.12.2-py3-none-any.whl
offline_packages/packaging_offline/uc_micro_py-1.0.3-py3-none-any.whl
parallel_options
pyproject.toml
src/forge/__init__.py
src/forge/__main__.py
src/forge/backend_manager.py
src/forge/backends/__init__.py
src/forge/backends/base.py
src/forge/backends/cp2k.py
src/forge/backends/elpa_selector.py
src/forge/backends/gpu_backend.py
src/forge/backends/quantum_espresso/__init__.py
src/forge/backends/quantum_espresso/backend.py
src/forge/backends/quantum_espresso/config_generator.py
src/forge/backends/quantum_espresso/executor.py
src/forge/backends/quantum_espresso/parser.py
src/forge/backends/vasp.py
src/forge/backends/wien2k/__init__.py
src/forge/backends/wien2k/core.py
src/forge/backends/wien2k/parsers.py
src/forge/benchmark/__init__.py
src/forge/benchmark/real.py
src/forge/benchmark/report.py
src/forge/benchmark/synthetic.py
src/forge/cli.py
src/forge/cli_commands/__init__.py
src/forge/cli_commands/_utils.py
src/forge/cli_commands/advise.py
src/forge/cli_commands/analyze.py
src/forge/cli_commands/analyze_bands.py
src/forge/cli_commands/base.py
src/forge/cli_commands/benchmark.py
src/forge/cli_commands/calibrate.py
src/forge/cli_commands/converge.py
src/forge/cli_commands/diagnose.py
src/forge/cli_commands/diagnostics.py
src/forge/cli_commands/generate.py
src/forge/cli_commands/hardware.py
src/forge/cli_commands/history.py
src/forge/cli_commands/monitor.py
src/forge/cli_commands/optimize.py
src/forge/cli_commands/predict.py
src/forge/cli_commands/run.py
src/forge/cli_commands/screen.py
src/forge/cli_commands/submit.py
src/forge/cli_commands/tui.py
src/forge/cli_commands/workflow.py
src/forge/cli_sbatch.py
src/forge/config.py
src/forge/core/__init__.py
src/forge/core/builder.py
src/forge/core/case_parser.py
src/forge/core/constants.py
src/forge/core/electronic_structure.py
src/forge/core/energy.py
src/forge/core/hardware/__init__.py
src/forge/core/hardware/cpu.py
src/forge/core/hardware/detection.py
src/forge/core/hardware/system.py
src/forge/core/hardware/types.py
src/forge/core/hardware/wrapper.py
src/forge/core/locator.py
src/forge/core/materials_project.py
src/forge/core/perf_counters.py
src/forge/core/pipeline.py
src/forge/core/scheduler.py
src/forge/core/terminal_monitor.py
src/forge/core/topology.py
src/forge/core/workflow.py
src/forge/core/workflow_executor.py
src/forge/exceptions.py
src/forge/logging_config.py
src/forge/ml/__init__.py
src/forge/ml/data_pipeline.py
src/forge/ml/gnn_kpoint_predictor.py
src/forge/ml/gnn_kpoint_v1.npz
src/forge/optimizer/__init__.py
src/forge/optimizer/advisor.py
src/forge/optimizer/bayesian/__init__.py
src/forge/optimizer/bayesian/acquisition.py
src/forge/optimizer/bayesian/bohb.py
src/forge/optimizer/bayesian/constraints.py
src/forge/optimizer/bayesian/core.py
src/forge/optimizer/bayesian/dpp.py
src/forge/optimizer/bayesian/elements.py
src/forge/optimizer/bayesian/gp.py
src/forge/optimizer/bayesian/kernels.py
src/forge/optimizer/bayesian/sampling.py
src/forge/optimizer/bayesian_tuner.py
src/forge/optimizer/convergence.py
src/forge/optimizer/gpu_detector.py
src/forge/optimizer/history.py
src/forge/optimizer/ml_predict.py
src/forge/optimizer/monitor/__init__.py
src/forge/optimizer/monitor/checkpoint.py
src/forge/optimizer/monitor/convergence.py
src/forge/optimizer/monitor/engine.py
src/forge/optimizer/monitor/types.py
src/forge/optimizer/parallel.py
src/forge/optimizer/profiler.py
src/forge/submit/__init__.py
src/forge/submit/lsf.py
src/forge/submit/pbs.py
src/forge/submit/slurm.py
src/forge/types.py
src/forge/ui/__init__.py
src/forge/ui/analysis.py
src/forge/ui/rich_ui.py
src/forge/utils/__init__.py
src/forge/utils/atomic_write.py
src/forge/utils/diagnostic.py
src/forge/utils/export.py
src/forge/utils/filelock.py
src/forge/utils/parallel_options.py
src/forge/utils/scratch.py
src/forge/utils/subprocess_utils.py
src/forge/utils/validation.py
src/forge/wizard.py
src/forge/wizard_sbatch.py
tests/__init__.py
tests/conftest.py
tests/fixtures/case_minimal.struct
tests/fixtures/hardware_avx512_ib.json
tests/fixtures/scf_converged.scf
tests/fixtures/scf_not_converged.scf
tests/fixtures/topology_slurm.json
tests/integration_test.py
tests/reframe/reframe_config.py
tests/reframe/wien2k_gen_test.py
tests/test_advisor.py
tests/test_analysis.py
tests/test_auto_detect.py
tests/test_bayesian.py
tests/test_bayesian_core.py
tests/test_bohb_dpp.py
tests/test_builder.py
tests/test_case_parser.py
tests/test_completions.py
tests/test_config.py
tests/test_convergence_opt.py
tests/test_diagnose.py
tests/test_elpa_selector.py
tests/test_gnn_kpoint_predictor.py
tests/test_hardware.py
tests/test_integration.py
tests/test_new_features.py
tests/test_parallel.py
tests/test_parallel_options.py
tests/test_perf_counters.py
tests/test_robustness.py
tests/test_scheduler.py
tests/test_slurm.py
tests/test_submit.py
tests/test_topology.py
tests/test_types.py
tests/test_utils.py
tests/test_wien2k_backend.py
tests/test_wien2k_flags.py
tests/test_wizard.py
```
