# Module Import Dependency Report

Revision: `c3db6e711acfbab1abe84b29f98d909374d0ab9b` (dirty).
Edges are unique production-code imports resolved from AST. Standard-library and third-party imports are summarized separately.

## Package-level dependencies

| From package | To package | Unique module edges |
|--------------|------------|--------------------:|
| `forge` | `forge.backend_manager` | 1 |
| `forge` | `forge.config` | 1 |
| `forge` | `forge.core` | 4 |
| `forge` | `forge.optimizer` | 3 |
| `forge` | `forge.types` | 1 |
| `forge` | `forge.utils` | 2 |
| `forge` | `forge.wizard` | 1 |
| `forge` | `forge.wizard_sbatch` | 1 |
| `forge.__main__` | `forge.cli` | 1 |
| `forge.backend_manager` | `forge.backends` | 1 |
| `forge.backend_manager` | `forge.types` | 1 |
| `forge.backends` | `forge.core` | 18 |
| `forge.backends` | `forge.exceptions` | 1 |
| `forge.backends` | `forge.logging_config` | 12 |
| `forge.backends` | `forge.optimizer` | 1 |
| `forge.backends` | `forge.types` | 3 |
| `forge.backends` | `forge.utils` | 4 |
| `forge.benchmark` | `forge.core` | 5 |
| `forge.benchmark` | `forge.logging_config` | 3 |
| `forge.benchmark` | `forge.submit` | 3 |
| `forge.benchmark` | `forge.ui` | 1 |
| `forge.benchmark` | `forge.utils` | 1 |
| `forge.cli` | `forge.backend_manager` | 1 |
| `forge.cli` | `forge.cli_commands` | 3 |
| `forge.cli` | `forge.config` | 1 |
| `forge.cli` | `forge.exceptions` | 1 |
| `forge.cli` | `forge.logging_config` | 1 |
| `forge.cli` | `forge.types` | 1 |
| `forge.cli` | `forge.ui` | 1 |
| `forge.cli_commands` | `forge.backend_manager` | 2 |
| `forge.cli_commands` | `forge.backends` | 1 |
| `forge.cli_commands` | `forge.benchmark` | 2 |
| `forge.cli_commands` | `forge.config` | 20 |
| `forge.cli_commands` | `forge.core` | 22 |
| `forge.cli_commands` | `forge.logging_config` | 2 |
| `forge.cli_commands` | `forge.optimizer` | 8 |
| `forge.cli_commands` | `forge.submit` | 2 |
| `forge.cli_commands` | `forge.types` | 1 |
| `forge.cli_commands` | `forge.ui` | 1 |
| `forge.cli_commands` | `forge.utils` | 2 |
| `forge.cli_sbatch` | `forge.backend_manager` | 1 |
| `forge.cli_sbatch` | `forge.config` | 1 |
| `forge.cli_sbatch` | `forge.core` | 1 |
| `forge.cli_sbatch` | `forge.exceptions` | 1 |
| `forge.cli_sbatch` | `forge.logging_config` | 1 |
| `forge.cli_sbatch` | `forge.submit` | 1 |
| `forge.cli_sbatch` | `forge.utils` | 1 |
| `forge.config` | `forge.core` | 1 |
| `forge.config` | `forge.types` | 1 |
| `forge.core` | `forge.backend_manager` | 2 |
| `forge.core` | `forge.config` | 1 |
| `forge.core` | `forge.exceptions` | 1 |
| `forge.core` | `forge.logging_config` | 17 |
| `forge.core` | `forge.optimizer` | 3 |
| `forge.core` | `forge.types` | 2 |
| `forge.core` | `forge.utils` | 5 |
| `forge.exceptions` | `forge.logging_config` | 1 |
| `forge.logging_config` | `forge.config` | 1 |
| `forge.logging_config` | `forge.exceptions` | 1 |
| `forge.ml` | `forge.logging_config` | 2 |
| `forge.ml` | `forge.optimizer` | 1 |
| `forge.optimizer` | `forge.backend_manager` | 3 |
| `forge.optimizer` | `forge.backends` | 2 |
| `forge.optimizer` | `forge.core` | 17 |
| `forge.optimizer` | `forge.logging_config` | 12 |
| `forge.optimizer` | `forge.utils` | 3 |
| `forge.submit` | `forge.core` | 4 |
| `forge.submit` | `forge.logging_config` | 3 |
| `forge.submit` | `forge.utils` | 1 |
| `forge.types` | `forge.core` | 1 |
| `forge.ui` | `forge.core` | 5 |
| `forge.ui` | `forge.logging_config` | 2 |
| `forge.ui` | `forge.optimizer` | 2 |
| `forge.ui` | `forge.submit` | 1 |
| `forge.ui` | `forge.utils` | 1 |
| `forge.utils` | `forge.core` | 4 |
| `forge.utils` | `forge.logging_config` | 8 |
| `forge.wizard` | `forge.backend_manager` | 1 |
| `forge.wizard` | `forge.core` | 4 |
| `forge.wizard` | `forge.exceptions` | 1 |
| `forge.wizard` | `forge.logging_config` | 1 |
| `forge.wizard` | `forge.optimizer` | 1 |
| `forge.wizard` | `forge.types` | 1 |
| `forge.wizard` | `forge.utils` | 1 |
| `forge.wizard_sbatch` | `forge.backend_manager` | 1 |
| `forge.wizard_sbatch` | `forge.core` | 1 |
| `forge.wizard_sbatch` | `forge.logging_config` | 1 |
| `forge.wizard_sbatch` | `forge.submit` | 2 |
| `forge.wizard_sbatch` | `forge.utils` | 1 |

## Highest in-degree (imported by many)

| Module | In | Out |
|--------|---:|----:|
| `forge.logging_config` | 66 | 2 |
| `forge.core.topology` | 27 | 1 |
| `forge.config` | 25 | 2 |
| `forge.core.hardware` | 21 | 3 |
| `forge.cli_commands.base` | 21 | 1 |
| `forge.cli_commands._utils` | 20 | 2 |
| `forge.utils.atomic_write` | 14 | 1 |
| `forge.core.scheduler` | 12 | 4 |
| `forge.backend_manager` | 12 | 2 |
| `forge.core.case_parser` | 12 | 2 |
| `forge.types` | 10 | 1 |
| `forge.optimizer.advisor` | 9 | 7 |
| `forge.core.locator` | 7 | 1 |
| `forge.core.constants` | 7 | 0 |
| `forge.exceptions` | 6 | 1 |

## Highest out-degree (imports many)

| Module | Out | In |
|--------|----:|--:|
| `forge` | 14 | 0 |
| `forge.optimizer.bayesian.core` | 12 | 2 |
| `forge.backends.wien2k.core` | 10 | 1 |
| `forge.core` | 10 | 0 |
| `forge.wizard` | 10 | 1 |
| `forge.benchmark.real` | 9 | 2 |
| `forge.cli` | 9 | 1 |
| `forge.cli_commands.advise` | 9 | 0 |
| `forge.cli_commands.generate` | 9 | 0 |
| `forge.cli_commands.submit` | 9 | 0 |
| `forge.core.pipeline` | 9 | 5 |
| `forge.optimizer.bayesian` | 9 | 3 |
| `forge.cli_commands.hardware` | 8 | 0 |
| `forge.core.builder` | 8 | 5 |
| `forge.optimizer.profiler` | 8 | 1 |

## Circular import components

Computed on unique production import edges **excluding** `type_checking` context. Function-local imports still count, because they execute if that function runs.

1. `forge.config`, `forge.core.hardware`, `forge.core.hardware.cpu`, `forge.core.hardware.detection`, `forge.core.hardware.system`, `forge.core.hardware.types`, `forge.core.hardware.wrapper`, `forge.core.locator`, `forge.exceptions`, `forge.logging_config`, `forge.types`
2. `forge.backend_manager`, `forge.backends`, `forge.backends.wien2k`, `forge.backends.wien2k.core`, `forge.optimizer.advisor`
3. `forge.core.workflow_executor`, `forge.optimizer`, `forge.optimizer.bayesian`, `forge.optimizer.bayesian.core`, `forge.optimizer.monitor`, `forge.optimizer.monitor.engine`

Edges in those components:

| From | To | Context | Location |
|------|----|---------|----------|
| `forge.backend_manager` | `forge.backends` | top_level | `src/forge/backend_manager.py`:10 |
| `forge.backends` | `forge.backends.wien2k` | conditional_function | `src/forge/backends/__init__.py`:114 |
| `forge.backends.wien2k` | `forge.backends.wien2k.core` | top_level | `src/forge/backends/wien2k/__init__.py`:3 |
| `forge.backends.wien2k.core` | `forge.optimizer.advisor` | conditional_function | `src/forge/backends/wien2k/core.py`:278 |
| `forge.config` | `forge.core.locator` | function | `src/forge/config.py`:224 |
| `forge.config` | `forge.types` | function | `src/forge/config.py`:257 |
| `forge.core.hardware` | `forge.core.hardware.detection` | top_level | `src/forge/core/hardware/__init__.py`:3 |
| `forge.core.hardware` | `forge.core.hardware.types` | top_level | `src/forge/core/hardware/__init__.py`:4 |
| `forge.core.hardware` | `forge.core.hardware.wrapper` | top_level | `src/forge/core/hardware/__init__.py`:9 |
| `forge.core.hardware.cpu` | `forge.logging_config` | top_level | `src/forge/core/hardware/cpu.py`:10 |
| `forge.core.hardware.detection` | `forge.core.hardware.cpu` | top_level | `src/forge/core/hardware/detection.py`:3 |
| `forge.core.hardware.detection` | `forge.core.hardware.system` | top_level | `src/forge/core/hardware/detection.py`:4 |
| `forge.core.hardware.detection` | `forge.core.hardware.types` | top_level | `src/forge/core/hardware/detection.py`:5 |
| `forge.core.hardware.system` | `forge.logging_config` | top_level | `src/forge/core/hardware/system.py`:11 |
| `forge.core.hardware.system` | `forge.core.hardware.types` | top_level | `src/forge/core/hardware/system.py`:12 |
| `forge.core.hardware.system` | `forge.core.locator` | function | `src/forge/core/hardware/system.py`:462 |
| `forge.core.hardware.types` | `forge.logging_config` | top_level | `src/forge/core/hardware/types.py`:8 |
| `forge.core.hardware.wrapper` | `forge.logging_config` | top_level | `src/forge/core/hardware/wrapper.py`:6 |
| `forge.core.hardware.wrapper` | `forge.core.hardware.detection` | top_level | `src/forge/core/hardware/wrapper.py`:7 |
| `forge.core.hardware.wrapper` | `forge.core.hardware.types` | top_level | `src/forge/core/hardware/wrapper.py`:8 |
| `forge.core.locator` | `forge.logging_config` | top_level | `src/forge/core/locator.py`:12 |
| `forge.core.workflow_executor` | `forge.optimizer.monitor` | conditional_function | `src/forge/core/workflow_executor.py`:511 |
| `forge.exceptions` | `forge.logging_config` | function | `src/forge/exceptions.py`:340 |
| `forge.logging_config` | `forge.exceptions` | function | `src/forge/logging_config.py`:70 |
| `forge.logging_config` | `forge.config` | function | `src/forge/logging_config.py`:230 |
| `forge.optimizer` | `forge.optimizer.bayesian` | top_level | `src/forge/optimizer/__init__.py`:23 |
| `forge.optimizer` | `forge.optimizer.monitor` | top_level | `src/forge/optimizer/__init__.py`:36 |
| `forge.optimizer.advisor` | `forge.backend_manager` | conditional_function | `src/forge/optimizer/advisor.py`:43 |
| `forge.optimizer.bayesian` | `forge.optimizer.bayesian.core` | top_level | `src/forge/optimizer/bayesian/__init__.py`:14 |
| `forge.optimizer.bayesian.core` | `forge.core.workflow_executor` | function | `src/forge/optimizer/bayesian/core.py`:116 |
| `forge.optimizer.monitor` | `forge.optimizer.monitor.engine` | top_level | `src/forge/optimizer/monitor/__init__.py`:22 |
| `forge.optimizer.monitor.engine` | `forge.optimizer` | conditional_function | `src/forge/optimizer/monitor/engine.py`:323 |
| `forge.types` | `forge.core.hardware` | conditional_function | `src/forge/types.py`:276 |

Self-imports:
- `forge.cli_commands`

## Isolated production modules

None at module level (every production module has at least one internal import edge).

## Isolated packages (top two name segments, cross-package edges only)

None. Packages such as `forge.utils` / `forge.ml` import other `forge.*` packages.

## Third-party imports (name counts)

- `rich` (57)
- `numpy` (17)
- `matplotlib` (3)
- `yaml` (3)
- `sklearn` (2)
- `requests` (1)
- `tqdm` (1)
- `filelock` (1)
- `tomli_w` (1)
- `h5py` (1)

## Standard library imports (top)

- `typing` (95)
- `pathlib` (68)
- `os` (44)
- `dataclasses` (38)
- `re` (35)
- `time` (34)
- `subprocess` (31)
- `__future__` (30)
- `json` (29)
- `shutil` (27)
- `math` (26)
- `argparse` (21)
- `contextlib` (19)
- `datetime` (11)
- `threading` (10)
- `signal` (10)
- `sys` (9)
- `logging` (7)
- `enum` (7)
- `hashlib` (5)
- `shlex` (4)
- `collections` (4)
- `tempfile` (4)
- `abc` (3)
- `sqlite3` (3)
- `functools` (2)
- `urllib` (2)
- `array` (2)
- `uuid` (2)
- `csv` (2)
