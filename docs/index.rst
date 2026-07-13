FORGE Documentation
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user-guide
   examples
   parallel-modes
   troubleshooting
   contributing
   api

Production-grade WIEN2k parallel configuration file generator and HPC job dispatcher
for density functional theory (DFT) calculations.

Key Features
------------

* **Multi-backend support**: WIEN2k, VASP, Quantum ESPRESSO, CP2K
* **Roofline model optimization**: Hardware-aware resource allocation based on
  memory bandwidth and computational intensity
* **SLURM/PBS/LSF integration**: Automatic scheduler detection and job script generation
* **NUMA-aware topology detection**: Full awareness of cache hierarchy, vector ISA,
  and interconnect fabric
* **SCF convergence monitoring**: Real-time detection of convergence stalls,
  charge sloshing, and Broyden mixing anomalies
* **Bayesian optimization**: Historical execution profiling with ML-based parameter tuning
* **Interactive TUI**: Textual-based terminal user interface for cluster environments
* **Air-gapped HPC support**: Offline installation for secure computing environments

Installation
------------

.. code-block:: bash

   pip install forge

Or from source:

.. code-block:: bash

   git clone https://github.com/majidhaddad02/forge
   cd forge
   make install

Quick Start
-----------

.. code-block:: bash

   export WIENROOT=/opt/codes/WIEN2k/v24.1
   cd /path/to/wien2k_case
   forge generate
   forge tui

Citation
--------

If you use FORGE in your research, please cite:

.. code-block:: bibtex

   @software{forge,
     author = {Haddad, Majid and Jalali Asadabadi, Saed},
     title = {FORGE: Production-Grade Parallel Configuration for HPC DFT},
     year = {2025},
      version = {0.1.0},
     url = {https://github.com/majidhaddad02/forge}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
