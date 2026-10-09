# Python module catalog

One-line descriptions from the module docstring first line, or empty if none. Based on implementation files, not directory names.

| Module | Path | Category | Classes | Functions | Docstring |
|--------|------|----------|--------:|----------:|-----------|
| `docs.conf` | `docs/conf.py` | documentation | 0 | 0 |  |
| `download_and_train` | `download_and_train.py` | script | 0 | 0 | One-shot script: download MP dataset, train GNN, save model. |
| `forge` | `src/forge/__init__.py` | production | 0 | 2 | FORGE - Production-Grade Parallel Configuration & HPC Job Dispatcher. |
| `forge.__main__` | `src/forge/__main__.py` | production | 0 | 0 | Entry point for `python -m forge`. |
| `forge.backend_manager` | `src/forge/backend_manager.py` | production | 1 | 8 | Backward-compatible re-export shim. Delegates to forge.backends (unified registry). |
| `forge.backends` | `src/forge/backends/__init__.py` | production | 1 | 14 | Unified backend registry, factory, and instance manager for DFT codes. |
| `forge.backends.base` | `src/forge/backends/base.py` | production | 9 | 18 | Abstract Base Class & Type Contracts for DFT Backend Implementations. |
| `forge.backends.cp2k` | `src/forge/backends/cp2k.py` | production | 1 | 19 | CP2K Backend - Production-Grade Configuration Generator for HPC Clusters. |
| `forge.backends.elpa_selector` | `src/forge/backends/elpa_selector.py` | production | 1 | 4 | Automatic Eigenvalue Solver Selection for DFT Diagonalization. |
| `forge.backends.gpu_backend` | `src/forge/backends/gpu_backend.py` | production | 1 | 14 | GPU-Aware DFT Execution Support for WIEN2k, VASP, and Quantum ESPRESSO. |
| `forge.backends.quantum_espresso` | `src/forge/backends/quantum_espresso/__init__.py` | production | 0 | 0 | Quantum ESPRESSO Backend Package Initialization. |
| `forge.backends.quantum_espresso.backend` | `src/forge/backends/quantum_espresso/backend.py` | production | 3 | 14 | Quantum ESPRESSO Backend - Production-Grade Configuration Generator for HPC Clusters. |
| `forge.backends.quantum_espresso.config_generator` | `src/forge/backends/quantum_espresso/config_generator.py` | production | 0 | 5 | Optimal Parallel Configuration Generator for Quantum ESPRESSO (QE). |
| `forge.backends.quantum_espresso.executor` | `src/forge/backends/quantum_espresso/executor.py` | production | 1 | 4 | executor.py - Quantum ESPRESSO Process Executor |
| `forge.backends.quantum_espresso.parser` | `src/forge/backends/quantum_espresso/parser.py` | production | 2 | 8 | parser.py - Quantum ESPRESSO Output Parser |
| `forge.backends.vasp` | `src/forge/backends/vasp.py` | production | 1 | 11 | VASP Backend - Production-Grade Configuration Generator for HPC Clusters. |
| `forge.backends.wien2k` | `src/forge/backends/wien2k/__init__.py` | production | 0 | 0 | WIEN2k DFT backend — production-grade parser and execution logic. |
| `forge.backends.wien2k.core` | `src/forge/backends/wien2k/core.py` | production | 1 | 34 |  |
| `forge.backends.wien2k.parsers` | `src/forge/backends/wien2k/parsers.py` | production | 2 | 7 | WIEN2k file parsers — extract problem parameters from DFT input/output files. |
| `forge.benchmark` | `src/forge/benchmark/__init__.py` | production | 0 | 0 | Benchmark Package Initialization for FORGE. |
| `forge.benchmark.real` | `src/forge/benchmark/real.py` | production | 4 | 10 | Real-World Benchmark Execution & Empirical Data Collection Module. |
| `forge.benchmark.report` | `src/forge/benchmark/report.py` | production | 3 | 13 | Benchmark Report Generator for WIEN2k Gen. |
| `forge.benchmark.synthetic` | `src/forge/benchmark/synthetic.py` | production | 5 | 13 | Synthetic Benchmark & Workload Simulation Module. |
| `forge.cli` | `src/forge/cli.py` | production | 0 | 2 | Command-Line Interface (CLI) Entry Point for FORGE. |
| `forge.cli_commands` | `src/forge/cli_commands/__init__.py` | production | 0 | 1 | CLI command auto-discovery and registration. |
| `forge.cli_commands._utils` | `src/forge/cli_commands/_utils.py` | production | 0 | 5 | Shared utilities and global state for CLI command modules. |
| `forge.cli_commands.advise` | `src/forge/cli_commands/advise.py` | production | 0 | 5 |  |
| `forge.cli_commands.analyze` | `src/forge/cli_commands/analyze.py` | production | 0 | 2 |  |
| `forge.cli_commands.analyze_bands` | `src/forge/cli_commands/analyze_bands.py` | production | 0 | 2 |  |
| `forge.cli_commands.base` | `src/forge/cli_commands/base.py` | production | 1 | 4 | Command base protocol and registry for CLI subcommands. |
| `forge.cli_commands.benchmark` | `src/forge/cli_commands/benchmark.py` | production | 0 | 2 |  |
| `forge.cli_commands.calibrate` | `src/forge/cli_commands/calibrate.py` | production | 0 | 2 |  |
| `forge.cli_commands.converge` | `src/forge/cli_commands/converge.py` | production | 0 | 2 |  |
| `forge.cli_commands.diagnose` | `src/forge/cli_commands/diagnose.py` | production | 0 | 2 |  |
| `forge.cli_commands.diagnostics` | `src/forge/cli_commands/diagnostics.py` | production | 0 | 2 |  |
| `forge.cli_commands.generate` | `src/forge/cli_commands/generate.py` | production | 0 | 2 |  |
| `forge.cli_commands.hardware` | `src/forge/cli_commands/hardware.py` | production | 0 | 2 |  |
| `forge.cli_commands.history` | `src/forge/cli_commands/history.py` | production | 0 | 2 |  |
| `forge.cli_commands.monitor` | `src/forge/cli_commands/monitor.py` | production | 0 | 2 |  |
| `forge.cli_commands.optimize` | `src/forge/cli_commands/optimize.py` | production | 0 | 2 |  |
| `forge.cli_commands.predict` | `src/forge/cli_commands/predict.py` | production | 0 | 2 |  |
| `forge.cli_commands.run` | `src/forge/cli_commands/run.py` | production | 0 | 2 |  |
| `forge.cli_commands.screen` | `src/forge/cli_commands/screen.py` | production | 0 | 2 |  |
| `forge.cli_commands.submit` | `src/forge/cli_commands/submit.py` | production | 0 | 11 |  |
| `forge.cli_commands.tui` | `src/forge/cli_commands/tui.py` | production | 0 | 2 |  |
| `forge.cli_commands.workflow` | `src/forge/cli_commands/workflow.py` | production | 0 | 2 |  |
| `forge.cli_sbatch` | `src/forge/cli_sbatch.py` | production | 0 | 9 | Specialized CLI Module for SLURM SBATCH Script Generation & Validation. |
| `forge.config` | `src/forge/config.py` | production | 2 | 16 | Central Configuration & Environment Management Module for FORGE. |
| `forge.core` | `src/forge/core/__init__.py` | production | 0 | 0 | Core Package Initialization for WIEN2k Parallel Execution Engine. |
| `forge.core.builder` | `src/forge/core/builder.py` | production | 3 | 13 | Generic Configuration Builder - Orchestrates DFT backend configuration generation. |
| `forge.core.case_parser` | `src/forge/core/case_parser.py` | production | 4 | 30 | WIEN2k Input File Parser (case.in1, case.in2, case.inm, etc.) |
| `forge.core.constants` | `src/forge/core/constants.py` | production | 0 | 0 | Physical and computational constants used throughout the forge package. |
| `forge.core.electronic_structure` | `src/forge/core/electronic_structure.py` | production | 0 | 10 | Electronic Structure Post-Processing Module for WIEN2k Output Files. |
| `forge.core.energy` | `src/forge/core/energy.py` | production | 2 | 16 | Energy Measurement Module — Intel RAPL / AMD APM / NVIDIA NVML Integration. |
| `forge.core.hardware` | `src/forge/core/hardware/__init__.py` | production | 0 | 0 | Hardware detection package — CPU, memory, NUMA, IO, interconnect detection. |
| `forge.core.hardware.cpu` | `src/forge/core/hardware/cpu.py` | production | 1 | 12 | CPU detection mixin: architecture, generation, cores, frequency, vector ISA, GFLOPS. |
| `forge.core.hardware.detection` | `src/forge/core/hardware/detection.py` | production | 1 | 0 | SysFSHardwareInfo: concrete hardware detection implementation. |
| `forge.core.hardware.system` | `src/forge/core/hardware/system.py` | production | 1 | 18 | System detection mixin: memory, NUMA, IO, interconnect, libraries, hardware profile. |
| `forge.core.hardware.types` | `src/forge/core/hardware/types.py` | production | 5 | 26 | Hardware detection types, TypedDicts, ABC, and utility parsers. |
| `forge.core.hardware.wrapper` | `src/forge/core/hardware/wrapper.py` | production | 0 | 27 | Module-level cached wrappers and provider management for hardware detection. |
| `forge.core.locator` | `src/forge/core/locator.py` | production | 0 | 3 | Centralized external dependency locator for FORGE. |
| `forge.core.materials_project` | `src/forge/core/materials_project.py` | production | 3 | 11 | Materials Project Integration for High-Throughput WIEN2k Screening. |
| `forge.core.perf_counters` | `src/forge/core/perf_counters.py` | production | 2 | 45 | Hardware Performance Counter Integration Module. |
| `forge.core.pipeline` | `src/forge/core/pipeline.py` | production | 3 | 7 | Unified Execution Pipeline for WIEN2k Configuration. |
| `forge.core.scheduler` | `src/forge/core/scheduler.py` | production | 1 | 15 | Environment Detection & Topology Scaling Module. |
| `forge.core.terminal_monitor` | `src/forge/core/terminal_monitor.py` | production | 4 | 8 | Live SCF monitor with Rich terminal display for HPC workflows. |
| `forge.core.topology` | `src/forge/core/topology.py` | production | 7 | 45 | Topology Data Model for HPC Resource Allocation & Parallel Execution Planning. |
| `forge.core.workflow` | `src/forge/core/workflow.py` | production | 4 | 34 | Workflow Provenance System for WIEN2k Computational Pipelines. |
| `forge.core.workflow_executor` | `src/forge/core/workflow_executor.py` | production | 3 | 24 | WorkflowExecutor — DAG runtime engine for automated WIEN2k pipelines. |
| `forge.exceptions` | `src/forge/exceptions.py` | production | 19 | 27 | Centralized Exception Hierarchy & Error Context Manager for FORGE. |
| `forge.logging_config` | `src/forge/logging_config.py` | production | 4 | 15 | Centralized Logging & Structured Output Engine for FORGE. |
| `forge.ml` | `src/forge/ml/__init__.py` | production | 0 | 0 | Machine learning subpackage for FORGE. |
| `forge.ml.data_pipeline` | `src/forge/ml/data_pipeline.py` | production | 1 | 6 | Materials Project data pipeline for GNN k-point grid training. |
| `forge.ml.gnn_kpoint_predictor` | `src/forge/ml/gnn_kpoint_predictor.py` | production | 3 | 27 | Crystal Graph Neural Network for K-point Grid Prediction. |
| `forge.optimizer` | `src/forge/optimizer/__init__.py` | production | 0 | 0 | Optimizer module for WIEN2k parallel configuration. |
| `forge.optimizer.advisor` | `src/forge/optimizer/advisor.py` | production | 5 | 19 | Expert Advisor Module with Scientific Scaling Models & Multi-Objective Optimization. |
| `forge.optimizer.bayesian` | `src/forge/optimizer/bayesian/__init__.py` | production | 0 | 0 | Bayesian optimization for WIEN2k SCF parameters. |
| `forge.optimizer.bayesian.acquisition` | `src/forge/optimizer/bayesian/acquisition.py` | production | 0 | 2 | Acquisition functions: Expected Improvement and q-EI. |
| `forge.optimizer.bayesian.bohb` | `src/forge/optimizer/bayesian/bohb.py` | production | 1 | 22 | BOHB: Bayesian Optimization Hyperband for WIEN2k multi-fidelity tuning. |
| `forge.optimizer.bayesian.constraints` | `src/forge/optimizer/bayesian/constraints.py` | production | 0 | 3 | Physics-based constraint estimation for Bayesian optimization. |
| `forge.optimizer.bayesian.core` | `src/forge/optimizer/bayesian/core.py` | production | 2 | 28 | Core Bayesian optimizer: orchestrator, warm-start, and main optimization loop. |
| `forge.optimizer.bayesian.dpp` | `src/forge/optimizer/bayesian/dpp.py` | production | 1 | 4 | DPP Batch Selection for diverse BO batches. |
| `forge.optimizer.bayesian.elements` | `src/forge/optimizer/bayesian/elements.py` | production | 0 | 5 | Periodic table data and chemical similarity for transfer learning. |
| `forge.optimizer.bayesian.gp` | `src/forge/optimizer/bayesian/gp.py` | production | 2 | 7 | Gaussian Process regression with ARD kernel optimisation. |
| `forge.optimizer.bayesian.kernels` | `src/forge/optimizer/bayesian/kernels.py` | production | 0 | 3 | Kernel functions for Gaussian Process regression. |
| `forge.optimizer.bayesian.sampling` | `src/forge/optimizer/bayesian/sampling.py` | production | 0 | 3 | Latin Hypercube Sampling and parameter encoding/decoding. |
| `forge.optimizer.bayesian_tuner` | `src/forge/optimizer/bayesian_tuner.py` | production | 2 | 12 | Bayesian Auto-Tuner for WIEN2k Parameters. |
| `forge.optimizer.convergence` | `src/forge/optimizer/convergence.py` | production | 1 | 18 | Automated Convergence Testing for WIEN2k Calculations. |
| `forge.optimizer.gpu_detector` | `src/forge/optimizer/gpu_detector.py` | production | 2 | 7 | GPU Offloading Detection for WIEN2k HPC Workflows. |
| `forge.optimizer.history` | `src/forge/optimizer/history.py` | production | 2 | 22 | SQLite-Based Execution History Store for Tracking & Learning from Past Runs. |
| `forge.optimizer.ml_predict` | `src/forge/optimizer/ml_predict.py` | production | 5 | 19 | ML-Based SCF Convergence Prediction for WIEN2k. |
| `forge.optimizer.monitor` | `src/forge/optimizer/monitor/__init__.py` | production | 0 | 0 | SCF monitoring: real-time convergence tracking, charge sloshing detection, and checkpoint management. |
| `forge.optimizer.monitor.checkpoint` | `src/forge/optimizer/monitor/checkpoint.py` | production | 0 | 6 | SCF checkpoint management: save, restore, incremental, cleanup. |
| `forge.optimizer.monitor.convergence` | `src/forge/optimizer/monitor/convergence.py` | production | 0 | 9 | SCF convergence diagnosis: charge sloshing, Broyden, Anderson, DIIS, FFT analysis. |
| `forge.optimizer.monitor.engine` | `src/forge/optimizer/monitor/engine.py` | production | 0 | 12 | Monitoring engine: real-time SCF loop, rebuild triggers, preemption, threading. |
| `forge.optimizer.monitor.types` | `src/forge/optimizer/monitor/types.py` | production | 4 | 7 | Monitor data types, enums, and shared global state. |
| `forge.optimizer.parallel` | `src/forge/optimizer/parallel.py` | production | 1 | 19 | NUMA-Aware Parallelization Engine for WIEN2k HPC Workflows. |
| `forge.optimizer.profiler` | `src/forge/optimizer/profiler.py` | production | 3 | 19 | Automatic Profiling of Parallel Configurations with Statistical Rigor & HPC Resilience. |
| `forge.submit` | `src/forge/submit/__init__.py` | production | 0 | 0 | Job Submission & Scheduler Integration Package. |
| `forge.submit.lsf` | `src/forge/submit/lsf.py` | production | 4 | 21 | LSF Job Submission Provider - Production-Grade Integration for IBM Spectrum LSF. |
| `forge.submit.pbs` | `src/forge/submit/pbs.py` | production | 3 | 15 | PBS/Torque Job Submission Provider - Production-Grade Integration for PBS Pro & Torque. |
| `forge.submit.slurm` | `src/forge/submit/slurm.py` | production | 3 | 12 | SLURM Job Submission & Advanced Script Generator Module. |
| `forge.types` | `src/forge/types.py` | production | 14 | 16 | Central Type Definitions & Data Contracts for FORGE. |
| `forge.ui` | `src/forge/ui/__init__.py` | production | 0 | 0 | UI Package — Terminal output formatting and SCF log analysis. |
| `forge.ui.analysis` | `src/forge/ui/analysis.py` | production | 3 | 9 | SCF Log Analysis & Performance Scaling Engine for HPC/DFT Workflows. |
| `forge.ui.rich_ui` | `src/forge/ui/rich_ui.py` | production | 4 | 28 | Rich CLI Fallback & Terminal Output Engine for FORGE. |
| `forge.utils` | `src/forge/utils/__init__.py` | production | 0 | 0 | Utility Package Initialization for HPC/DFT Workflows. |
| `forge.utils.atomic_write` | `src/forge/utils/atomic_write.py` | production | 0 | 3 | Atomic File Write Utility Module. |
| `forge.utils.diagnostic` | `src/forge/utils/diagnostic.py` | production | 2 | 14 | System & Environment Diagnostic Module for HPC/DFT Workflows. |
| `forge.utils.export` | `src/forge/utils/export.py` | production | 3 | 13 | Export & Serialization Utility Module for HPC/DFT Workflows. |
| `forge.utils.filelock` | `src/forge/utils/filelock.py` | production | 3 | 11 | Process-Safe File Locking Utility for HPC & Distributed Workflows. |
| `forge.utils.parallel_options` | `src/forge/utils/parallel_options.py` | production | 1 | 6 | WIEN2k Parallel Options Generator & Parser Module. |
| `forge.utils.scratch` | `src/forge/utils/scratch.py` | production | 2 | 12 | HPC Scratch Space Management & Multi-Node I/O Staging Module. |
| `forge.utils.subprocess_utils` | `src/forge/utils/subprocess_utils.py` | production | 1 | 9 | Subprocess Execution & Process Lifecycle Management Module. |
| `forge.utils.validation` | `src/forge/utils/validation.py` | production | 2 | 7 | Configuration Validation & Backup Management Module for HPC/DFT Workflows. |
| `forge.wizard` | `src/forge/wizard.py` | production | 0 | 5 | Interactive CLI Configuration Wizard for FORGE. |
| `forge.wizard_sbatch` | `src/forge/wizard_sbatch.py` | production | 8 | 15 | Interactive Job Submission Wizard for FORGE. |
| `tests` | `tests/__init__.py` | test | 0 | 0 | Test Package Initialization for FORGE. |
| `tests.codebase_map.test_extractor` | `tests/codebase_map/test_extractor.py` | test | 1 | 9 | Tests for tools/codebase_map (stdlib-only AST analysis). |
| `tests.conftest` | `tests/conftest.py` | test | 0 | 7 | Central Pytest Configuration & Shared Fixtures. |
| `tests.integration_test` | `tests/integration_test.py` | test | 5 | 12 | Integration Tests for FORGE. |
| `tests.reframe.reframe_config` | `tests/reframe/reframe_config.py` | test | 0 | 0 | Minimal ReFrame Configuration for WIEN2kGen CI/CD Testing. |
| `tests.reframe.wien2k_gen_test` | `tests/reframe/wien2k_gen_test.py` | test | 6 | 22 | ReFrame Tests for WIEN2kGen — HPC Configuration Generator. |
| `tests.test_advisor` | `tests/test_advisor.py` | test | 6 | 35 | Production-Grade Tests for optimizer.advisor Module. |
| `tests.test_analysis` | `tests/test_analysis.py` | test | 4 | 25 | Tests for ui/analysis.py — SCF Log Parsing & Parallel Scaling Analysis. |
| `tests.test_auto_detect` | `tests/test_auto_detect.py` | test | 3 | 10 | Production-Grade Tests for Backend Auto-Detection & Manager Integration. |
| `tests.test_bayesian` | `tests/test_bayesian.py` | test | 10 | 50 | Production-Grade Tests for optimizer.bayesian Module. |
| `tests.test_bayesian_core` | `tests/test_bayesian_core.py` | test | 2 | 9 | Tests for Bayesian search-space physics constraints in define_search_space. |
| `tests.test_bohb_dpp` | `tests/test_bohb_dpp.py` | test | 3 | 20 | Tests for BOHB (TPE/KDE) and DPP batch selector. |
| `tests.test_builder` | `tests/test_builder.py` | test | 0 | 4 | Tests for forge.core.builder.build_auto suggestion-validation flow. |
| `tests.test_case_parser` | `tests/test_case_parser.py` | test | 0 | 49 | Tests for forge.core.case_parser — WIEN2k input file parsing. |
| `tests.test_completions` | `tests/test_completions.py` | test | 0 | 8 | Shell completion scripts must complete CLI subcommands without bash-completion. |
| `tests.test_config` | `tests/test_config.py` | test | 3 | 7 | Production-Grade Tests for config.py Module. |
| `tests.test_convergence_opt` | `tests/test_convergence_opt.py` | test | 13 | 82 | Tests for forge.optimizer.convergence — SCF divergence detection, |
| `tests.test_diagnose` | `tests/test_diagnose.py` | test | 1 | 12 | Tests for forge diagnose — SCF parser flags and healthy-status gating. |
| `tests.test_elpa_selector` | `tests/test_elpa_selector.py` | test | 2 | 7 | Tests for forge.backends.elpa_selector — solver routing and compile flags. |
| `tests.test_gnn_kpoint_predictor` | `tests/test_gnn_kpoint_predictor.py` | test | 2 | 10 | Tests for GNN k-point predictor — analytical gradient verification. |
| `tests.test_hardware` | `tests/test_hardware.py` | test | 10 | 42 | Production-Grade Tests for core.hardware Module. |
| `tests.test_integration` | `tests/test_integration.py` | test | 2 | 7 | Integration tests — end-to-end pipeline with realistic WIEN2k case directories. |
| `tests.test_new_features` | `tests/test_new_features.py` | test | 0 | 12 | Tests for new features: SGE, MPICH/MVAPICH, --reserve-os-cores, system type, CPU gen. |
| `tests.test_parallel` | `tests/test_parallel.py` | test | 20 | 122 | Tests for forge.optimizer.parallel — NUMA-aware parallelization engine. |
| `tests.test_parallel_options` | `tests/test_parallel_options.py` | test | 0 | 17 | Tests for parallel_options generation and the _write_parallel_options method. |
| `tests.test_perf_counters` | `tests/test_perf_counters.py` | test | 1 | 27 | Unit tests for hardware counter detection and load-driven measurement. |
| `tests.test_robustness` | `tests/test_robustness.py` | test | 4 | 16 | Robustness tests — bad inputs, missing files, edge cases. |
| `tests.test_scheduler` | `tests/test_scheduler.py` | test | 2 | 10 | Production-Grade Tests for core.scheduler Module. |
| `tests.test_slurm` | `tests/test_slurm.py` | test | 0 | 10 | Tests for SLURM memory-unit parsing and mail-directive defaults. |
| `tests.test_submit` | `tests/test_submit.py` | test | 5 | 26 | Tests for forge submit resource resolution from .machines allocation. |
| `tests.test_topology` | `tests/test_topology.py` | test | 8 | 52 | Production-Grade Tests for core.topology Module. |
| `tests.test_types` | `tests/test_types.py` | test | 4 | 22 | Tests for Wien2kFlags, CalculationType, and ExecutionMode enums. |
| `tests.test_utils` | `tests/test_utils.py` | test | 3 | 14 | Production-Grade Tests for utils/ Module. |
| `tests.test_wien2k_backend` | `tests/test_wien2k_backend.py` | test | 25 | 183 | Comprehensive tests for WIEN2k backend (core.py) and parsers (parsers.py). |
| `tests.test_wien2k_flags` | `tests/test_wien2k_flags.py` | test | 0 | 16 | Integration tests for WIEN2k flag detection and execution command generation. |
| `tests.test_wizard` | `tests/test_wizard.py` | test | 3 | 12 | Tests for wizard.py — WIENROOT detection, validation, and scratch health. |
| `tools.codebase_map` | `tools/codebase_map/__init__.py` | tool | 0 | 0 | Static codebase mapping toolkit for FORGE / wien2k_gen. |
| `tools.codebase_map.analyze` | `tools/codebase_map/analyze.py` | tool | 0 | 14 | Static architecture analysis for FORGE / wien2k_gen. |
| `tools.codebase_map.extractor` | `tools/codebase_map/extractor.py` | tool | 7 | 32 | AST-based static extractor for FORGE / wien2k_gen. |
| `tools.codebase_map.gen_graphs` | `tools/codebase_map/gen_graphs.py` | tool | 0 | 8 | DOT and Mermaid graph emitters. |
| `tools.codebase_map.gen_inventory` | `tools/codebase_map/gen_inventory.py` | tool | 0 | 1 | File tree and inventory markdown. |
| `tools.codebase_map.gen_markdown` | `tools/codebase_map/gen_markdown.py` | tool | 0 | 10 | Human-readable architecture reports. |
| `tools.codebase_map.reports` | `tools/codebase_map/reports.py` | tool | 0 | 1 | Write all human and diagram artifacts under docs/architecture/. |
| `tools.codebase_map.stdlib_modules` | `tools/codebase_map/stdlib_modules.py` | tool | 0 | 1 | Python 3.9–3.12 standard-library module names used for import classification. |
| `tools.codebase_map.textutil` | `tools/codebase_map/textutil.py` | tool | 0 | 2 | Shared text helpers for architecture reports. |
