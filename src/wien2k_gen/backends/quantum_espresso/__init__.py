"""
Quantum ESPRESSO Backend Package Initialization.
Exports core components for QE input parsing, parallel configuration generation,
execution management, and output analysis. Designed for seamless integration
with the wien2k_gen multi-DFT optimization pipeline.

Submodules:
• backend: Main Backend implementation for QE 6.7/7.x+
• config_generator: Advanced domain decomposition optimizer (npool, ndiag, nband, ntg)
• parser: Robust output log parser for energies, forces, convergence, and timing
• executor: HPC-grade process launcher with environment injection & preemption handling
"""

# =============================================================================
# Core Backend & API Exports
# =============================================================================

# Main Backend class
from .backend import QuantumEspressoBackend

# Configuration & Domain Decomposition Optimizer
from .config_generator import (
    generate_qe_config,
    optimal_nband,
    optimal_npool,
)

# Execution & Process Management
from .executor import (
    execute_qe_calculation,
)

# Output Analysis & Parsing
from .parser import (
    parse_qe_output,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================

# Controls `from wien2k_gen.backends.quantum_espresso import *`
# and provides clear IDE auto-completion boundaries.
__all__ = [
    "QuantumEspressoBackend",
    "execute_qe_calculation",
    "generate_qe_config",
    "optimal_nband",
    "optimal_npool",
    "parse_qe_output",
]