"""SCF convergence diagnosis: charge sloshing, Broyden, Anderson, DIIS, FFT analysis."""

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .types import ConvergenceAnalysis

from ...logging_config import get_logger

logger = get_logger(__name__)


def detect_charge_sloshing(dayfile_content: str, tolerance: float = 0.5) -> dict:
    """
    Detect charge sloshing—oscillatory convergence behavior where charge distance
    alternates (high/low) across SCF cycles instead of decaying monotonically.

    Uses the Durbin-Watson statistic on consecutive charge-distance deltas to
    identify negative autocorrelation, the hallmark signature of sloshing.

    Args:
        dayfile_content: Raw or lowercased text content from the dayfile/SCF log.
        tolerance: Sensitivity threshold for DW-based detection (lower = more sensitive).

    Returns:
        dict with keys:
            sloshing_detected: bool
            severity: float (0.0 - 1.0, based on oscillation amplitude)
            cycles_affected: List[int] of cycle indices exhibiting sloshing
            recommendation: str with actionable mixing advice
    """
    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    if len(matches) < 5:
        return {
            "sloshing_detected": False,
            "severity": 0.0,
            "cycles_affected": [],
            "recommendation": ""
        }

    values = [float(m) for m in matches]
    n = len(values)
    diffs = [values[i] - values[i - 1] for i in range(1, n)]

    numerator = sum(d * d for d in diffs)
    denominator = sum((v - sum(values) / n) ** 2 for v in values)
    dw_stat = 2.0 if denominator < 1e-30 else numerator / denominator if denominator > 0 else 2.0

    sloshing_detected = dw_stat > (2.5 + tolerance)

    slope = values[-1] - values[0]
    if sloshing_detected and slope > 0:
        sloshing_detected = False

    mean_val = sum(values) / n if n > 0 else 0
    rel_range = (max(values) - min(values)) / mean_val if mean_val > 1e-12 else 0
    severity_raw = min(1.0, (dw_stat - 2.5) / 1.5)
    severity = round(0.8 * severity_raw + 0.2 * min(1.0, rel_range), 4)

    cycles_affected: list[int] = []
    if sloshing_detected:
        osc_threshold = (max(diffs) - min(diffs)) * 0.3 if diffs else 0
        for i in range(2, n):
            if diffs[i - 1] * diffs[i - 2] < 0 and abs(diffs[i - 1]) > osc_threshold * 0.1:
                cycles_affected.append(i + 1)

    return {
        "sloshing_detected": sloshing_detected,
        "severity": severity,
        "cycles_affected": cycles_affected,
        "recommendation": (
            "Charge sloshing detected—reduce mixing beta by 50%, "
            "increase number of PRATT iterations, or try MSR1a mixing"
            if sloshing_detected
            else ""
        )
    }


_SLOSHING_REMEDIATION = {
    "metallic": [
        {"action": "set_kerker_mixing", "params": {"q0": 0.1, "beta": 0.10},
         "reason": "Kerker preconditioned mixing suppresses long-wavelength charge sloshing in metals"},
        {"action": "set_smearing", "params": {"type": "Methfessel-Paxton", "width_ry": 0.02},
         "reason": "Metallic systems require Fermi surface smearing to stabilize SCF"},
        {"action": "set_mixing", "params": {"beta": 0.10, "method": "PRATT"},
         "reason": "Backup: PRATT mixing with beta=0.10 if Kerker fails"},
        {"action": "increase_kpoints", "params": {"factor": 2.0},
         "reason": "Denser k-mesh captures Fermi surface better"},
    ],
    "symmetry_breaking": [
        {"action": "disable_symmetry", "params": {},
         "reason": "Switch to runsp_lapw to handle broken symmetry"},
        {"action": "set_mixing", "params": {"beta": 0.05},
         "reason": "Conservative mixing for symmetry-broken systems"},
        {"action": "set_rmt", "params": {"reduction": 0.95},
         "reason": "Slightly reduce RMT to avoid sphere overlap"},
    ],
    "core_overlap": [
        {"action": "check_rmt_ratios", "params": {},
         "reason": "RMT ratio > 1.5 between atoms; check sphere overlap"},
        {"action": "adjust_r0", "params": {"reduction": 0.90},
         "reason": "Reduce R0 to separate core from valence spheres"},
        {"action": "set_mixing", "params": {"beta": 0.02},
         "reason": "Very conservative mixing for core overlap cases"},
        {"action": "increase_pratt", "params": {"cycles": 5},
         "reason": "Increase PRATT iterations to 5 cycles"},
    ],
    "mixing_too_aggressive": [
        {"action": "set_mixing", "params": {"beta": 0.05},
         "reason": "Reduce mixing beta by 50% from current value"},
        {"action": "increase_pratt", "params": {"cycles": 3},
         "reason": "Increase PRATT iterations for better preconditioning"},
        {"action": "try_msr1a", "params": {},
         "reason": "Switch to multi-secant MSR1a mixing for improved stability"},
    ],
}


def _build_sloshing_remediation(root_cause: str, severity: float) -> list[dict]:
    """Build ordered remediation steps scaled by severity."""
    template = _SLOSHING_REMEDIATION.get(root_cause, _SLOSHING_REMEDIATION["mixing_too_aggressive"])
    actions = []
    for step in template:
        action = dict(step)
        action["priority"] = "critical" if severity > 0.7 else "high"
        actions.append(action)
    return actions


def diagnose_charge_sloshing_root_cause(  # noqa: C901
    dayfile_content: str,
    case_name: str = "case",
    struct_path: Optional[str] = None,
) -> dict:
    """Diagnose root cause of charge sloshing and recommend targeted fix.

    Based on WIEN2k Usersguide 2023 §4.5 (SCF convergence diagnostics):
    charge sloshing has specific root causes that require different
    treatments:

    1. Metallic systems (Fermi surface, bands crossing EF):
       → Apply Methfessel-Paxton smearing (0.02 Ry), PRATT mixing
    2. Symmetry breaking (structural distortion, magnetic ordering):
       → Disable symmetry (runsp_lapw), reduce mixing slowly
    3. Core overlap (RMT ratio < 1.5 between adjacent atoms):
       → Check RMT values, adjust R0, increase PRATT iterations
    4. Default (mixing rate too aggressive):
       → Reduce mixing beta, increase PRATT cycles, try MSR1a

    Returns dict with:
        root_cause: str — one of "metallic", "symmetry_breaking",
                   "core_overlap", "mixing_too_aggressive", "unknown"
        actions: List[dict] — ordered list of remediation steps
        confidence: float — 0.0-1.0 diagnostic confidence
    """
    slosh = detect_charge_sloshing(dayfile_content)
    if not slosh["sloshing_detected"]:
        return {
            "root_cause": "none",
            "actions": [],
            "confidence": 0.0,
        }

    indicators: dict[str, float] = {}

    # 1. Check for metallic system indicators
    if case_name:
        try:
            scf_path = Path(f"{case_name}.scf")
            if not scf_path.exists():
                scf_path = Path(case_name) / f"{case_name}.scf"
            if scf_path.exists():
                scf_text = scf_path.read_text()
                if ":GAP" in scf_text:
                    gap_match = re.search(r':GAP\s*:\s*(-?\d+\.\d+)', scf_text)
                    if gap_match:
                        gap = float(gap_match.group(1))
                        if gap < 0.01:
                            indicators["metallic"] = 0.9
                        elif gap < 0.1:
                            indicators["metallic"] = 0.6
                if "FERMI" in scf_text or "nearly free" in scf_text.lower():
                    indicators.setdefault("metallic", 0.0)
                    indicators["metallic"] = max(indicators["metallic"], 0.5)
        except Exception:
            logger.debug("SCF text analysis failed", exc_info=True)

    # 2. Check for symmetry breaking
    if ("symmetry" in dayfile_content.lower() or "symm" in dayfile_content.lower()) and ("broken" in dayfile_content.lower() or "fail" in dayfile_content.lower()):
        indicators["symmetry_breaking"] = 0.85

    # 3. Check for core overlap via RMT info
    rmt_pattern = re.compile(r'(?:RMT|rmt)\s*[=:]\s*([\d]+\.?\d*)', re.IGNORECASE)
    rmt_matches = rmt_pattern.findall(dayfile_content)
    if len(rmt_matches) >= 2:
        rmt_vals = [float(v) for v in rmt_matches]
        min_rmt = min(rmt_vals)
        max_rmt = max(rmt_vals)
        if min_rmt > 0 and max_rmt / min_rmt > 1.5:
            indicators["core_overlap"] = min(0.9, (max_rmt / min_rmt - 1.5) * 2)

    # 4. Check case.inc for mixing settings
    if case_name:
        try:
            inc_path = Path(f"{case_name}.inc")
            if not inc_path.exists():
                inc_path = Path(case_name) / f"{case_name}.inc"
            if inc_path.exists():
                inc_text = inc_path.read_text()
                mix_match = re.search(r'([\d]+\.?\d*)\s*\n', inc_text)
                if mix_match:
                    beta = float(mix_match.group(1))
                    if beta > 0.3:
                        indicators["mixing_too_aggressive"] = 0.8
                    elif beta > 0.2:
                        indicators["mixing_too_aggressive"] = 0.4
        except Exception:
            logger.debug("Mixing parameter extraction failed", exc_info=True)

    # 5. Default: if nothing specific, assume aggressive mixing
    if not indicators:
        indicators["mixing_too_aggressive"] = 0.5

    root_cause = max(indicators, key=indicators.get)
    confidence = indicators[root_cause]

    actions = _build_sloshing_remediation(root_cause, slosh["severity"])

    return {
        "root_cause": root_cause,
        "confidence": round(confidence, 2),
        "indicators": indicators,
        "actions": actions,
    }


def detect_charge_sloshing_fft(dayfile_content: str) -> dict:
    """
    Detect charge sloshing via frequency-domain (FFT) analysis.

    Per Kresse & Furthmueller 1996 (PRB 54, 11169), charge sloshing manifests
    as high-frequency oscillations in the charge density residual. While the
    Durbin-Watson statistic catches alternating patterns, FFT-based analysis
    identifies the dominant frequency components explicitly.

    This function computes the power spectral density (PSD) of the charge
    distance time series using the discrete Fourier transform. A strong
    peak at the Nyquist frequency (period ≈ 2 SCF cycles) is the hallmark
    of charge sloshing. The normalised high-frequency power ratio quantifies
    how much of the total spectral energy resides above the median frequency.

    Requires at least 8 SCF cycles for meaningful frequency resolution.

    Returns:
        dict with keys:
            sloshing_detected: bool
            dominant_frequency_hz: float (in cycles per SCF iteration)
            hf_power_ratio: float (0.0-1.0, ratio of power above median freq)
            recommendation: str
    """
    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    if len(matches) < 8:
        return {
            "sloshing_detected": False,
            "dominant_frequency_hz": 0.0,
            "hf_power_ratio": 0.0,
            "recommendation": "Insufficient cycles for FFT analysis (need >= 8)"
        }

    values = [float(m) for m in matches]
    n = len(values)

    demeaned = [v - sum(values) / n for v in values]
    if n % 2 == 1:
        demeaned = demeaned[:-1]
        n = len(demeaned)

    try:
        fft_result = [abs(x) ** 2 for x in _simple_dft(demeaned)]
    except Exception:
        logger.debug("FFT periodicity detection failed", exc_info=True)
        return {
            "sloshing_detected": False,
            "dominant_frequency_hz": 0.0,
            "hf_power_ratio": 0.0,
            "recommendation": "FFT computation failed"
        }

    pos_spectrum = fft_result[1:n // 2 + 1]
    if not pos_spectrum or sum(pos_spectrum) < 1e-30:
        return {
            "sloshing_detected": False,
            "dominant_frequency_hz": 0.0,
            "hf_power_ratio": 0.0,
            "recommendation": "Flat spectrum: no oscillatory behaviour detected"
        }

    total_power = sum(pos_spectrum)
    median_idx = (n // 2) // 2
    if median_idx < 1:
        median_idx = 1
    hf_power = sum(pos_spectrum[median_idx:])
    hf_ratio = hf_power / total_power if total_power > 0 else 0.0

    max_idx = max(range(len(pos_spectrum)), key=lambda i: pos_spectrum[i])
    dominant_freq = (max_idx + 1) / n

    sloshing_detected = (
        hf_ratio > 0.4 and dominant_freq > 0.3
    )

    recommendation = ""
    if sloshing_detected:
        recommendation = (
            f"FFT confirms charge sloshing (HF ratio={hf_ratio:.2f}, "
            f"dominant freq={dominant_freq:.3f}/cycle). "
            f"Per Kresse & Furthmueller 1996: increase Kerker "
            f"preconditioning, reduce mixing beta, or use PRATT mixing."
        )

    return {
        "sloshing_detected": sloshing_detected,
        "dominant_frequency_hz": round(dominant_freq, 4),
        "hf_power_ratio": round(hf_ratio, 4),
        "recommendation": recommendation
    }


def _simple_dft(signal: list[float]) -> list[complex]:
    """
    Simple discrete Fourier transform (DFT) for short time series.
    Uses O(N^2) algorithm suitable for SCF cycle counts (N < 200).
    For production use with long time series, numpy.fft is preferred.
    """
    n = len(signal)
    result = []
    for k in range(n):
        real = 0.0
        imag = 0.0
        for t in range(n):
            angle = 2.0 * math.pi * k * t / n
            real += signal[t] * math.cos(angle)
            imag -= signal[t] * math.sin(angle)
        result.append(complex(real, imag))
    return result


def analyze_broyden_mixing(dayfile_content: str, log_path: str) -> dict:
    """
    Analyze Broyden mixing behavior and detect stagnation.

    Checks whether Broyden mixing is active, examines charge-distance
    progression across iterations, and identifies plateauing that signals
    a stuck Broyden history.

    Args:
        dayfile_content: Text content from the dayfile/SCF log.
        log_path: Path to the log file (reserved for extended diagnostics).

    Returns:
        dict with keys:
            broyden_active: bool
            stuck: bool
            convergence_rate: float (log-average reduction per cycle)
            iteration_plateau_length: int
            recommendation: str
    """
    broyden_active = bool(
        re.search(r'\bbroy', dayfile_content, re.IGNORECASE)
    )

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    values = [float(m) for m in matches]

    n = len(values)
    convergence_rate = 0.0
    if n >= 3:
        ratios = []
        for i in range(1, n):
            if values[i - 1] > 1e-12:
                ratios.append(math.log(values[i] / values[i - 1]))
        if ratios:
            convergence_rate = round(math.exp(sum(ratios) / len(ratios)), 6)

    plateau_length = 0
    stuck = False
    if n >= 4:
        rel_tol = 0.02
        run = 1
        max_run = 1
        for i in range(1, n):
            ref = max(values[i - 1], 1e-12)
            if abs(values[i] - values[i - 1]) / ref < rel_tol:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        plateau_length = max_run
        stuck = plateau_length >= 3 and any(v > 1e-6 for v in values[-plateau_length:])

    recommendation = ""
    if broyden_active and stuck:
        recommendation = (
            "Broyden mixing stuck—reset Broyden history, "
            "increase mixing parameter, fall back to PRATT mixing, "
            "or try Anderson mixing (faster for metallic systems, Eyert 1996)"
        )

    return {
        "broyden_active": broyden_active,
        "stuck": stuck,
        "convergence_rate": convergence_rate,
        "iteration_plateau_length": plateau_length,
        "recommendation": recommendation
    }


def analyze_anderson_mixing(dayfile_content: str) -> dict:
    """
    Analyze Anderson mixing behavior and detect stagnation.

    Anderson mixing (Eyert 1996, J. Comp. Phys. 124, 271) generalizes Broyden
    by mixing a linear combination of previous charge densities with coefficients
    chosen to minimise the residual. For metallic systems, Anderson often
    converges 2-3x faster than Broyden because it constructs a better
    approximate inverse Jacobian from the charge density history.

    This detector checks for Anderson-specific signatures:
    1. ``:MIX`` lines indicating Anderson or extended Anderson mixing
    2. Charge distance plateauing despite active Anderson iterations
    3. Recommendations to switch to Broyden (insulators) or PRATT (sloshing)

    Returns:
        dict with keys:
            anderson_active: bool
            stuck: bool
            convergence_rate: float
            iteration_plateau_length: int
            recommendation: str
    """
    anderson_active = bool(
        re.search(r'(?:\banderson\b|:MIX\s*[12])', dayfile_content, re.IGNORECASE)
    )

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    values = [float(m) for m in matches]
    n = len(values)

    convergence_rate = 0.0
    if n >= 3:
        ratios = []
        for i in range(1, n):
            if values[i - 1] > 1e-12:
                ratios.append(math.log(values[i] / values[i - 1]))
        if ratios:
            convergence_rate = round(math.exp(sum(ratios) / len(ratios)), 6)

    plateau_length = 0
    stuck = False
    if n >= 4:
        rel_tol = 0.01
        run = 1
        max_run = 1
        for i in range(1, n):
            ref = max(values[i - 1], 1e-12)
            if abs(values[i] - values[i - 1]) / ref < rel_tol:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        plateau_length = max_run
        stuck = plateau_length >= 4 and any(v > 1e-6 for v in values[-plateau_length:])

    recommendation = ""
    if anderson_active and stuck:
        recommendation = (
            "Anderson mixing stuck—try tightening the mixing parameter, "
            "increase mixing history depth, or switch to Broyden mixing "
            "(better Jacobian approximation for insulators)"
        )

    return {
        "anderson_active": anderson_active,
        "stuck": stuck,
        "convergence_rate": convergence_rate,
        "iteration_plateau_length": plateau_length,
        "recommendation": recommendation
    }


def analyze_diis_mixing(dayfile_content: str) -> dict:
    """
    Analyze DIIS/Pulay mixing behavior and detect divergence.

    DIIS (Direct Inversion in the Iterative Subspace), also known as Pulay
    mixing (Pulay 1980, Chem. Phys. Lett. 73, 393), constructs an optimal
    linear combination of Fock matrices by minimising the error vector norm
    in a Krylov subspace. For charge-density mixing in DFT (Kresse &
    Furthmueller 1996, PRB 54, 11169), DIIS converges quadratically near
    the minimum but can diverge catastrophically if the initial guess is
    poor or the subspace becomes linearly dependent.

    Detector checks:
    1. ``:DIIS`` or ``:PULAY`` keywords in the mixing log
    2. Charge distance INCREASING over recent iterations (divergence)
    3. Residual norm oscillation without convergence

    Returns:
        dict with keys:
            diis_active: bool
            diverging: bool
            residual_trend: float (+ = diverging, - = converging)
            recommendation: str
    """
    diis_active = bool(
        re.search(r'(?:\bdiis\b|\:DIIS|\bpulay\b|:PULAY)', dayfile_content, re.IGNORECASE)
    )

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    res_pattern = re.compile(
        r'(?:residual|error)\s*(?:norm)?\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )

    values = [float(m) for m in cd_pattern.findall(dayfile_content)]
    _residuals = [float(m) for m in res_pattern.findall(dayfile_content)]
    n = len(values)

    residual_trend = 0.0
    if n >= 5:
        recent = values[-5:]
        window_len = len(recent)
        x_mean = sum(range(window_len)) / window_len
        y_mean = sum(recent) / window_len
        num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(window_len))
        den = sum((i - x_mean) ** 2 for i in range(window_len))
        if den > 1e-12:
            residual_trend = num / den / (y_mean + 1e-12)

    diverging = False
    if n >= 5:
        recent_vals = values[-5:]
        max_recent = max(recent_vals)
        early_vals = values[:min(5, n // 2)]
        avg_early = sum(early_vals) / len(early_vals) if early_vals else 0
        if avg_early > 1e-12 and (max_recent / avg_early) > 2.0:
            diverging = True
        if n >= 10:
            last_half_avg = sum(values[n // 2:]) / (n - n // 2)
            first_half_avg = sum(values[:n // 2]) / (n // 2)
            if first_half_avg > 1e-12 and last_half_avg / first_half_avg > 1.5:
                diverging = True

    recommendation = ""
    if diis_active and diverging:
        recommendation = (
            "DIIS/Pulay mixing diverging—reduce mixing history depth "
            "(NDII=5 or less), increase Kerker preconditioning, "
            "or fall back to Broyden mixing with smaller beta"
        )
    elif diis_active and not diverging and residual_trend > 0.0 and residual_trend < 0.1:
        recommendation = (
            "DIIS convergence slow; consider increasing DIIS history "
            "dimension or switching to Anderson mixing for metallic systems"
        )

    return {
        "diis_active": diis_active,
        "diverging": diverging,
        "residual_trend": round(residual_trend, 6),
        "recommendation": recommendation
    }


def analyze_convergence_history(dayfile_content: str) -> 'ConvergenceAnalysis':  # noqa: C901
    """
    Read full convergence history from SCF log content and classify the
    convergence trajectory, providing mixing recommendations and cycle estimates.

    Args:
        dayfile_content: Text content from the dayfile/SCF log.

    Returns:
        ConvergenceAnalysis dataclass with classification, history vectors, and advice.
    """
    from .types import ConvergenceAnalysis

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    en_pattern = re.compile(
        r'(?:(?:total|:)?\s*energy)\s*[=:]\s*([\-]?[\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )

    charge_distances = [float(m) for m in cd_pattern.findall(dayfile_content)]
    energies = [float(m) for m in en_pattern.findall(dayfile_content)]

    n = len(charge_distances)
    if n < 3:
        return ConvergenceAnalysis(
            convergence_type="unknown",
            mixing_recommendation="Insufficient data for convergence classification",
            estimated_cycles_to_converge=-1,
            charge_distance_history=charge_distances,
            energy_history=energies
        )

    slope = charge_distances[-1] - charge_distances[0]
    ratios = []
    for i in range(1, n):
        if charge_distances[i - 1] > 1e-12:
            ratios.append(charge_distances[i] / charge_distances[i - 1])

    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

    diffs = [charge_distances[i] - charge_distances[i - 1] for i in range(1, n)]
    sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
    oscillation_ratio = sign_changes / max(1, len(diffs) - 1) if len(diffs) > 1 else 0.0

    rel_plateau = 0
    if n >= 4:
        rel_tol = 0.02
        run = 1
        for i in range(1, n):
            ref = max(charge_distances[i - 1], 1e-12)
            if abs(charge_distances[i] - charge_distances[i - 1]) / ref < rel_tol:
                run += 1
                rel_plateau = max(rel_plateau, run)
            else:
                run = 1

    if avg_ratio > 1.01 and slope > 0:
        convergence_type = "divergent"
        recommendation = (
            "Divergent SCF—reduce mixing beta, enable PRATT mixing, "
            "or add more k-points to stabilise charge density"
        )
        est_cycles = -1
    elif rel_plateau >= 3 and charge_distances[-1] > 1e-5:
        convergence_type = "stalled"
        recommendation = (
            "SCF stalled—increase mixing iterations, "
            "enable Broyden mixing, or shake the density with a small displacement"
        )
        est_cycles = -1
    elif oscillation_ratio >= 0.4 and avg_ratio < 1.0:
        convergence_type = "oscillatory"
        recommendation = (
            "Oscillatory convergence (charge sloshing)—reduce mixing beta, "
            "increase PRATT iterations, or switch to MSR1a mixing"
        )
        est = 0
        if avg_ratio > 0 and avg_ratio < 1.0:
            est = int(math.log(1e-6 / max(charge_distances[-1], 1e-12))
                     / math.log(avg_ratio)) + 1
        est_cycles = max(0, est)
    elif avg_ratio < 1.0 and oscillation_ratio < 0.4:
        convergence_type = "monotonic"
        recommendation = "Convergence progressing monotonically—current settings are suitable"
        est = 0
        if avg_ratio > 0 and avg_ratio < 1.0:
            est = int(math.log(1e-6 / max(charge_distances[-1], 1e-12))
                     / math.log(avg_ratio)) + 1
        est_cycles = max(0, est)
    else:
        convergence_type = "unknown"
        recommendation = "Unclear convergence pattern—monitor further"
        est_cycles = -1

    return ConvergenceAnalysis(
        convergence_type=convergence_type,
        mixing_recommendation=recommendation,
        estimated_cycles_to_converge=est_cycles,
        charge_distance_history=charge_distances,
        energy_history=energies
    )


__all__ = [
    "analyze_anderson_mixing",
    "analyze_broyden_mixing",
    "analyze_convergence_history",
    "analyze_diis_mixing",
    "detect_charge_sloshing",
    "detect_charge_sloshing_fft",
    "diagnose_charge_sloshing_root_cause",
]
