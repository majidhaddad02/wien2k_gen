"""CPU detection mixin: architecture, generation, cores, frequency, vector ISA, GFLOPS."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

from ...logging_config import get_logger

logger = get_logger(__name__)


class CpuDetectionMixin:
    """CPU detection methods for SysFSHardwareInfo."""

    @staticmethod
    def _run_cmd_safe(cmd: list[str], timeout: int = 5, force_c_locale: bool = False) -> Optional[str]:
        """Safely execute a shell command with timeout and stderr suppression."""
        env = os.environ.copy()
        if force_c_locale:
            env['LC_ALL'] = 'C'
        try:
            return subprocess.check_output(
                cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL, env=env
            ).strip()
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            logger.debug(f"Command failed or not found: {' '.join(cmd)}")
            return None

    def _parse_lscpu_flags(self) -> list[str]:
        """Extract CPU flags from lscpu."""
        raw = self._run_cmd_safe(["lscpu"], force_c_locale=True)
        if not raw:
            return []
        for line in raw.split('\n'):
            if (line.strip().startswith("Flags:") or line.strip().startswith("CPU op-mode")) and ":" in line:
                return line.split(':', 1)[1].strip().split()
        return []

    def get_logical_cores(self) -> int:
        return os.cpu_count() or 1

    def get_physical_cores(self) -> int:  # noqa: C901
        raw = self._run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
        if raw:
            try:
                data = json.loads(raw)
                fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
                sockets = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
                cores_per_socket = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
                return sockets * cores_per_socket
            except (json.JSONDecodeError, ValueError, StopIteration):
                logger.debug("lscpu JSON parsing failed, trying fallback")

        raw = self._run_cmd_safe(["lscpu", "--parse=CPU,SOCKET,CORE"], force_c_locale=True)
        if raw:
            try:
                lines = [line for line in raw.split('\n') if line and not line.startswith('#')]
                unique_physical = set()
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3 and parts[1] != '-' and parts[2] != '-':
                        unique_physical.add((parts[1], parts[2]))
                if unique_physical:
                    return len(unique_physical)
            except Exception as e:
                logger.debug(f"lscpu parse-based core detection failed: {e}")

        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
            pairs = set()
            current_phys, current_core = None, None
            for line in content.split('\n'):
                if line.startswith('physical id'):
                    current_phys = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    current_core = line.split(':')[1].strip()
                if current_phys is not None and current_core is not None:
                    pairs.add((current_phys, current_core))
            if pairs:
                return len(pairs)

            core_ids = set(re.findall(r'^core id\s*:\s*(\d+)', content, re.MULTILINE))
            if core_ids:
                return len(core_ids)
        except Exception as e:
            logger.debug(f"/proc/cpuinfo parsing failed: {e}")

        return self.get_logical_cores()

    def is_hyperthreading_active(self) -> bool:
        return self.get_logical_cores() > self.get_physical_cores()

    def get_vector_isa_and_width(self) -> dict[str, Union[str, int]]:
        flags = self._parse_lscpu_flags()
        flag_set = set(flags)

        if any(f.startswith("avx512") for f in flag_set):
            return {"isa": "avx512", "width_bits": 512}
        if "avx2" in flag_set or "avx" in flag_set:
            return {"isa": "avx2", "width_bits": 256}
        if "sve" in flag_set or "sve2" in flag_set:
            return {"isa": "sve", "width_bits": 128}
        if "neon" in flag_set:
            return {"isa": "neon", "width_bits": 128}
        if "sse4_2" in flag_set or "sse4_1" in flag_set:
            return {"isa": "sse4", "width_bits": 128}

        return {"isa": "scalar", "width_bits": 64}

    def get_fma_units_per_core(self) -> int:
        vector_info = self.get_vector_isa_and_width()
        isa = str(vector_info["isa"])

        if "avx512" in isa:
            return 2
        if "avx" in isa or "avx2" in isa:
            return 2
        if "neon" in isa or "sve" in isa:
            return 1
        return 0

    def calculate_peak_fp64_gflops(self) -> float:
        raw = self._run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
        sockets = 1
        cores_per_socket = 1

        if raw:
            try:
                data = json.loads(raw)
                fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
                sockets = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
                cores_per_socket = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
            except Exception as e:
                logger.debug(f"lscpu JSON parsing failed in peak GFLOPS: {e}")

        freq_info = self.get_cpu_frequency_info()
        base_freq = freq_info.get("base", 2000.0)
        max_freq = freq_info.get("max", 0.0)

        # scaling_max_freq is the single-core turbo maximum. On multi-core
        # chips (e.g., EPYC 64-core) the all-core sustained frequency under
        # AVX load can be 20-40% lower. Use base_freq as the conservative
        # estimate and apply per-architecture throttle factors below.
        effective_freq = base_freq
        if effective_freq == 0.0:
            effective_freq = max(base_freq, max_freq)

        fma = self.get_fma_units_per_core()
        vec_width = int(self.get_vector_isa_and_width()["width_bits"])
        isa = str(self.get_vector_isa_and_width()["isa"])
        cpu_arch = self.get_cpu_architecture()

        ops_per_core_per_cycle = fma * (vec_width / 64.0) * 2.0

        # AVX frequency throttle table (Intel SDM / AMD PPR):
        # AVX-512 heavy workloads can downclock 10-25% on Intel Skylake-SP/Ice Lake
        # AVX2 downclock is ~5-10% on Intel, minimal on AMD Zen
        # Values: fraction of base frequency sustained under all-core AVX load
        # Ref: Intel 64 and IA-32 Architectures Optimization Reference Manual
        #      (Table "Intel AVX-512 Frequency Licenses"), AMD PPR §2.1.2
        # TODO: parameterize per-generation — SapphireRapids has lower AVX-512
        #       throttle (~0.80 per-core) vs Ice Lake (~0.85)
        if isa == "avx512":
            if "xeon" in cpu_arch:
                throttle_factor = 0.85
            elif "epyc" in cpu_arch:
                throttle_factor = 0.95
            else:
                throttle_factor = 0.90
        elif isa in ("avx2", "avx"):
            throttle_factor = 0.92 if "xeon" in cpu_arch else 0.97
        elif isa in ("sve", "neon"):
            throttle_factor = 0.95
        else:
            throttle_factor = 1.0

        adjusted_freq = effective_freq * throttle_factor

        peak = sockets * cores_per_socket * adjusted_freq * 1e6 * ops_per_core_per_cycle / 1e9

        return round(peak, 2)

    def get_cpu_governor(self, cpu_id: int = 0) -> Optional[str]:
        path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor"
        try:
            return Path(path).read_text().strip()
        except FileNotFoundError:
            return None

    def get_cpu_frequency_info(self) -> dict[str, float]:
        info = {"min": 0.0, "max": 0.0, "current": 0.0, "base": 0.0}
        base_path = Path("/sys/devices/system/cpu/cpu0/cpufreq")

        if base_path.exists():
            try:
                info["min"] = float((base_path / "scaling_min_freq").read_text().strip()) / 1000
                info["max"] = float((base_path / "scaling_max_freq").read_text().strip()) / 1000
                info["current"] = float((base_path / "scaling_cur_freq").read_text().strip()) / 1000
            except Exception as e:
                logger.debug(f"Frequency reading failed: {e}")

        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "cpu MHz" in line:
                        info["base"] = float(line.split(':')[1].strip())
                        break
        except Exception as e:
            logger.debug(f"cpuinfo frequency fallback: {e}")
            info["base"] = info["max"] if info["max"] > 0 else 2000.0

        return info

    def get_cpu_architecture(self) -> str:
        raw = self._run_cmd_safe(["lscpu"], force_c_locale=True)
        if raw:
            if "AMD" in raw:
                return "epyc" if "EPYC" in raw else "amd_ryzen"
            elif "Intel" in raw:
                if "Xeon" in raw:
                    # Check for hybrid (P-core + E-core) Xeon generations
                    # Intel Thread Director exposes heterogeneous topology
                    # via /sys/devices/system/cpu/types
                    if Path("/sys/devices/system/cpu/types").exists():
                        return "xeon_hybrid"
                    return "xeon"
                # Alder Lake / Raptor Lake: P-core + E-core
                if Path("/sys/devices/system/cpu/types").exists():
                    return "intel_hybrid"
                return "intel_consumer"
            elif "aarch64" in raw or "ARM" in raw:
                return "arm_neoverse" if "Neoverse" in raw else "arm"
        return "unknown"

    def get_cpu_generation(self) -> str:  # noqa: C901
        """
        Detect specific CPU generation from model name.

        Parses /proc/cpuinfo model name to identify:
        Intel: Xeon Platinum 8480+, Xeon Gold 6348, Xeon E5-2690v4, Core i9-13900K
        AMD:   EPYC 9654 (Genoa), EPYC 7763 (Milan), EPYC 7742 (Rome), EPYC 7501 (Naples)
        ARM:   Neoverse-N1, Neoverse-V1, Ampere Altra

        Returns a canonical string like "Xeon_SapphireRapids", "EPYC_Genoa", "Neoverse_N1"
        or "unknown" if parsing fails.
        """
        try:
            raw = self._run_cmd_safe(["lscpu"], force_c_locale=True)
            model_line = ""
            if raw:
                for line in raw.splitlines():
                    if "odel name" in line:
                        model_line = line.split(":", 1)[-1].strip()
                        break
            if not model_line:
                model_line = getattr(self, "_get_cpuinfo_model", lambda: "")() or ""

            model_lower = model_line.lower()

            arch = self.get_cpu_architecture()

            if arch in ("xeon", "intel_consumer"):
                if "platinum" in model_lower:
                    model_digits = "".join(c for c in model_line if c.isdigit())
                    if model_digits.startswith("86"):
                        return "Xeon_SierraForest"
                    if model_digits.startswith("85"):
                        return "Xeon_EmeraldRapids"
                    if model_digits.startswith("84"):
                        return "Xeon_SapphireRapids"
                    if model_digits.startswith("83"):
                        return "Xeon_IceLake"
                    if model_digits.startswith("82"):
                        return "Xeon_CascadeLake"
                    if model_digits.startswith("81"):
                        return "Xeon_Skylake"
                    return "Xeon_SapphireRapids"
                if "gold 6" in model_lower or "gold 5" in model_lower:
                    if "63" in model_line and "v" not in model_lower:
                        return "Xeon_SapphireRapids"
                    return "Xeon_IceLake"
                if "gold" in model_lower or "silver" in model_lower:
                    if "52" in model_line or "62" in model_line:
                        return "Xeon_CascadeLake"
                    if "51" in model_line or "61" in model_line:
                        return "Xeon_Skylake"
                    if "v4" in model_lower:
                        return "Xeon_Broadwell"
                    if "v3" in model_lower:
                        return "Xeon_Haswell"
                    if "v2" in model_lower:
                        return "Xeon_IvyBridge"
                    return "Xeon_Skylake"
                if "eon" in model_lower:
                    if "e5" in model_lower or "e7" in model_lower:
                        return "Xeon_SandyBridge" if "v1" in model_lower or "-2" in model_line else "Xeon_Haswell"
                    if "e3" in model_lower:
                        return "Xeon_CoffeeLake"
                    return "Xeon_Skylake"
                if "core" in model_lower and "ultra" in model_lower:
                    return "CoreUltra_MeteorLake"
                if "core" in model_lower:
                    if "13" in model_line or "14" in model_line:
                        return "Core_RaptorLake"
                    if "12" in model_line:
                        return "Core_AlderLake"
                    if "11" in model_line:
                        return "Core_TigerLake"
                    if "10" in model_line:
                        return "Core_IceLake"
                    return "Core_Consumer"
                if "i9" in model_lower or "i7" in model_lower or "i5" in model_lower:
                    gen_part = model_line.split("-")[-1][:2] if "-" in model_line else ""
                    if gen_part and gen_part.isdigit():
                        gen_num = int(gen_part)
                        if gen_num >= 14:
                            return "Core_RaptorLake"
                        if gen_num >= 12:
                            return "Core_AlderLake"
                    return "Core_Consumer"
                return arch

            if arch in ("epyc", "amd_ryzen"):
                if "epyc" in model_lower:
                    model_words = model_line.split()
                    first_num = ""
                    for w in model_words:
                        digits = "".join(c for c in w if c.isdigit())
                        if len(digits) >= 4:
                            first_num = digits[:4]
                            break
                    if first_num:
                        model_int = int(first_num[:4]) if len(first_num) >= 4 else 0
                        if model_int >= 9004:
                            return "EPYC_Genoa"
                        if model_int >= 8004:
                            return "EPYC_Siena"
                        if model_int >= 7004:
                            return "EPYC_Bergamo"
                        if model_int >= 7003:
                            return "EPYC_MilanX"
                        if model_int >= 7002:
                            return "EPYC_Rome"
                        if model_int >= 7001:
                            return "EPYC_Naples"
                    return "EPYC_Milan"
                if "ryzen" in model_lower:
                    if "9950" in model_line or "9900" in model_line:
                        return "Ryzen_GraniteRidge"
                    if "7950" in model_line or "7900" in model_line:
                        return "Ryzen_Raphael"
                    if "5950" in model_line or "5900" in model_line:
                        return "Ryzen_Vermeer"
                    return "Ryzen_Consumer"
                return arch

            elif "arm" in arch.lower() or "neoverse" in arch.lower():
                if "neoverse-v2" in model_lower:
                    return "Neoverse_V2"
                if "neoverse-v1" in model_lower:
                    return "Neoverse_V1"
                if "neoverse-n2" in model_lower:
                    return "Neoverse_N2"
                if "neoverse-n1" in model_lower:
                    return "Neoverse_N1"
                if "ampere" in model_lower:
                    return "Ampere_Altra"
                if "graviton" in model_lower:
                    if "4" in model_line:
                        return "Graviton4"
                    if "3" in model_line:
                        return "Graviton3"
                return "ARMv8"

            return "unknown"
        except Exception as e:
            logger.debug(f"CPU microarch detection failed: {e}")
            return "unknown"
