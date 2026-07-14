# Installation Guide

## Requirements

- **Python** >= 3.9
- **Linux** (x86_64 or aarch64) — full HPC features
- **macOS** — basic features only (no NUMA, no scheduler detection)
- **WIEN2k** installation with valid license

## Standard Installation (Recommended)

Using the bundled installer script:

```bash
chmod +x install.sh
./install.sh
```

This creates a virtual environment in `~/.local/opt/forge/`, installs the
package and its dependencies, and adds symlinks for `forge`, `forge_sbatch`,
and `forge_wizard` into `~/.local/bin/`.

Verify:
```bash
forge --version
```

### Installer Options

| Flag | Description |
|------|-------------|
| `--prefix=/custom/path` | Custom install location |
| `--dry-run` | Preview without installing |
| `--force` | Overwrite existing installation |
| `--uninstall` | Remove all traces |

### Root (System-Wide) Installation

```bash
sudo ./install.sh
```

Installs to `/opt/forge/` with symlinks in `/usr/local/bin/`.

## From Source (Developers)

```bash
git clone https://github.com/majidhaddad02/forge.git
cd forge
pip install -e ".[dev]"
```

Or use the Makefile:
```bash
make dev        # full dev install with lint/test tools
make install    # core only
make minimal    # essential deps only (no TUI)
```

## Air-Gapped HPC Installation

For clusters without internet access — use the installer's offline mode:

```bash
# On a machine with internet:
pip download -r requirements/core.txt -d ./offline_packages
rsync -av . cluster:/path/to/forge/

# On the cluster:
cd /path/to/forge
./install.sh
```

The installer auto-detects the `offline_packages/` directory and installs
from local wheels when no network is available.

## Container Deployment

### Docker
```bash
docker build -t forge .
docker run --rm -v $(pwd):/work forge generate
```

### Singularity / Apptainer
```bash
<!-- TODO: add container registry URL when published -->
singularity build forge.sif docker-daemon://forge:latest
singularity exec --bind $(pwd):/work forge.sif forge generate
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WIENROOT` | Path to WIEN2k installation | auto-detected |
| `SCRATCH` | Scratch directory | `/tmp` |
| `TMPDIR` | Temporary files | `$SCRATCH` |
| `LOG_LEVEL` | Python logging: DEBUG, INFO, WARNING, ERROR | INFO |
| `NO_COLOR` | Disable colored output | unset |

## Optional Dependencies

| Package | Needed For |
|---------|------------|
| `rich` | Colored CLI output (auto-installed) |
| `textual` | Interactive TUI (`forge tui`) |
| `numpy` | Roofline model calculations |
| `psutil` | Enhanced process monitoring |

## Verifying Installation

```bash
forge diagnostics
```

This runs a full hardware and environment diagnostic, displaying:
- CPU architecture and generation
- Physical vs logical cores, hyperthreading status
- NUMA topology
- Scheduler type
- MPI detection (vendor, version, launcher)
- WIEN2k installation path and version
- Scratch filesystem type
- Network interconnect (InfiniBand/Ethernet)
