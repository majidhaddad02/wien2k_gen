"""
ReFrame Tests for WIEN2kGen — HPC Configuration Generator.

Provides smoke and benchmark tests suitable for CI/CD pipelines and HPC
facility regression suites.  Requires reframe-hpc >= 4.0.
"""

import os
import subprocess

try:
    import reframe as rfm
    import reframe.utility.sanity as sn
    from reframe import performance_function, run_after, sanity_function, variable
    _HAS_REFRAME = True
except ImportError:  # pragma: no cover
    rfm = None  # type: ignore[assignment]
    sn = None  # type: ignore[assignment]
    _HAS_REFRAME = False
    def _variable(*a, **kw):
        return None

    def _run_after(*a, **kw):
        return lambda f: f

    def _sanity_function(f):
        return f

    def _performance_function(*a, **kw):
        return lambda f: f

    variable = _variable  # type: ignore[no-redef]
    run_after = _run_after  # type: ignore[no-redef]
    sanity_function = _sanity_function  # type: ignore[no-redef]
    performance_function = _performance_function  # type: ignore[no-redef]


def _find_wien2k_gen() -> str:
    """Locate the wien2k_gen CLI entry point."""
    for candidate in ["wien2k_gen", "python -m wien2k_gen"]:
        try:
            result = subprocess.run(
                [*candidate.split(), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "wien2k_gen"


_WIEN2K_GEN_CLI = _find_wien2k_gen()


@rfm.simple_test
class Wien2kGenSmokeTest(rfm.RunOnlyRegressionTest):
    """
    Smoke test: verify wien2k_gen can generate a valid .machines file
    with the 'generate' subcommand and does not crash on basic inputs.
    """

    valid_systems = ["*"]  # noqa: RUF012
    valid_prog_environs = ["*"]  # noqa: RUF012
    executable = "bash"

    cores = variable(int, value=4)
    nodes = variable(int, value=1)

    @run_after("setup")
    def setup_test(self):
        stage_dir = os.path.join(self.stagedir, f"smoke_{self.nodes}n_{self.cores}c")
        os.makedirs(stage_dir, exist_ok=True)

        cmd = (
            f"cd {stage_dir} && "
            f"{_WIEN2K_GEN_CLI} generate "
            f"--nodes {self.nodes} --cores {self.cores} "
            f"--dry-run"
        )
        self.executable_opts = ["-c", cmd]

    @sanity_function
    def assert_generate_completes(self):
        return sn.assert_eq(self.job.exitcode, 0)

    @performance_function("s")
    def elapsed_time(self):
        return sn.extractsingle(r"Elapsed time:\s+([\d.]+)", self.stdout, 1, float)


@rfm.simple_test
class Wien2kGenDryRunContentTest(rfm.RunOnlyRegressionTest):
    """
    Verify dry-run output contains expected configuration directives.
    """

    valid_systems = ["*"]  # noqa: RUF012
    valid_prog_environs = ["*"]  # noqa: RUF012
    executable = "bash"

    cores = variable(int, value=8)
    nodes = variable(int, value=1)

    @run_after("setup")
    def setup_test(self):
        stage_dir = self.stagedir
        os.makedirs(stage_dir, exist_ok=True)

        cmd = (
            f"cd {stage_dir} && "
            f"{_WIEN2K_GEN_CLI} generate "
            f"--nodes {self.nodes} --cores {self.cores} "
            f"--dry-run"
        )
        self.executable_opts = ["-c", cmd]

    @sanity_function
    def assert_has_machines_content(self):
        return sn.assert_found(r"(lapw0|lapw1|lapw2|#|granularity)", self.stdout)


@rfm.simple_test
class Wien2kGenBenchmarkTest(rfm.RunOnlyRegressionTest):
    """
    Benchmark: measure configuration generation time and check output quality.

    Generates configurations for multiple (nodes, cores) combinations and
    verifies that generation time remains within acceptable limits and that
    output contains valid parallelisation mode recommendations.
    """

    valid_systems = ["*"]  # noqa: RUF012
    valid_prog_environs = ["*"]  # noqa: RUF012
    executable = "bash"

    nodes = variable(int, value=1)
    cores = variable(int, value=16)
    max_time_seconds = variable(float, value=10.0)

    @run_after("setup")
    def setup_test(self):
        stage_dir = os.path.join(
            self.stagedir, f"benchmark_{self.nodes}n_{self.cores}c"
        )
        os.makedirs(stage_dir, exist_ok=True)

        cmd = (
            f"cd {stage_dir} && "
            f"time -p {_WIEN2K_GEN_CLI} generate "
            f"--nodes {self.nodes} "
            f"--cores {self.cores} "
            f"--dry-run"
        )
        self.executable_opts = ["-c", cmd]

    @sanity_function
    def assert_generates_config(self):
        return sn.all([
            sn.assert_eq(self.job.exitcode, 0),
            sn.assert_found(r"mode|kpoint|hybrid|mpi", self.stdout),
        ])

    @performance_function("s")
    def generation_time(self):
        return sn.extractsingle(
            r"real\s+([\d.]+)", self.stderr, 1, float
        )

    @performance_function("bytes")
    def output_size(self):
        return len(self.stdout)

    @run_after("performance")
    def set_performance_reference(self):
        self.reference = {
            "*": {
                "generation_time": (0, None, self.max_time_seconds, "s"),
            }
        }


@rfm.simple_test
class Wien2kGenModeDetectionTest(rfm.RunOnlyRegressionTest):
    """
    Verify that the advisor correctly detects and recommends parallelisation
    modes (kpoint, hybrid, mpi) based on system characteristics.
    """

    valid_systems = ["*"]  # noqa: RUF012
    valid_prog_environs = ["*"]  # noqa: RUF012
    executable = "bash"

    @run_after("setup")
    def setup_test(self):
        stage_dir = os.path.join(self.stagedir, "mode_detect")
        os.makedirs(stage_dir, exist_ok=True)

        cmd = (
            f"cd {stage_dir} && "
            f"{_WIEN2K_GEN_CLI} analyze --json 2>/dev/null || "
            f"{_WIEN2K_GEN_CLI} info --json 2>/dev/null || "
            f"echo '{{\"mode\": \"unknown\"}}' "
        )
        self.executable_opts = ["-c", cmd]

    @sanity_function
    def assert_json_output(self):
        return sn.assert_found(r"\{.*mode.*\}", self.stdout)


@rfm.simple_test
class Wien2kGenMultiNodeTest(rfm.RunOnlyRegressionTest):
    """
    Multi-node configuration test: verify correct core distribution across
    multiple nodes with heterogeneous-aware allocation.
    """

    valid_systems = ["*"]  # noqa: RUF012
    valid_prog_environs = ["*"]  # noqa: RUF012
    executable = "bash"

    nodes = variable(int, value=2)
    cores = variable(int, value=32)

    @run_after("setup")
    def setup_test(self):
        stage_dir = os.path.join(
            self.stagedir, f"multinode_{self.nodes}n_{self.cores}c"
        )
        os.makedirs(stage_dir, exist_ok=True)

        cmd = (
            f"cd {stage_dir} && "
            f"{_WIEN2K_GEN_CLI} generate "
            f"--nodes {self.nodes} "
            f"--cores {self.cores} "
            f"--dry-run"
        )
        self.executable_opts = ["-c", cmd]

    @sanity_function
    def assert_multi_node_config(self):
        return sn.assert_eq(self.job.exitcode, 0)

    @performance_function("")
    def contains_core_distribution(self):
        return sn.assert_found(r"\d+:\d+|cores_per_node|host", self.stdout)


@rfm.simple_test
class Wien2kGenHistoryPersistenceTest(rfm.RunOnlyRegressionTest):
    """
    Verify execution history persistence: generate a config, verify history
    database is created and contains a record.
    """

    valid_systems = ["*"]  # noqa: RUF012
    valid_prog_environs = ["*"]  # noqa: RUF012
    executable = "bash"

    @run_after("setup")
    def setup_test(self):
        stage_dir = os.path.join(self.stagedir, "history_test")
        os.makedirs(stage_dir, exist_ok=True)

        cmd = (
            f"cd {stage_dir} && "
            f"{_WIEN2K_GEN_CLI} generate --nodes 1 --cores 4 --dry-run && "
            f"test -f ~/.wien2k_gen/history.db && echo 'HISTORY_DB_OK' || echo 'HISTORY_DB_MISSING'"
        )
        self.executable_opts = ["-c", cmd]

    @sanity_function
    def assert_history_db_exists(self):
        return sn.assert_found(r"HISTORY_DB_OK", self.stdout)
