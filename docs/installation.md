# Installation Guide

## Requirements

- **Python** >= 3.9 with the `venv` module (`python3-venv` on Debian/Ubuntu)
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
and `forge_wizard` into `~/.local/bin/`. The installer can be run from any
working directory; it locates the repository from its own path.

If core libraries such as NumPy, PyYAML, or Rich are already installed on
the selected Python, the installer reuses them via `--system-site-packages`
instead of downloading copies. Pass `--no-system-packages` to isolate the
venv completely.

If FORGE is already installed at the chosen prefix, the installer asks
whether to **update** (keep the venv, install only missing/outdated
packages) or **reinstall** (delete the previous tree and install from
scratch). Non-interactive defaults:
```bash
./install.sh --yes --update      # keep venv, upgrade package
./install.sh --yes --reinstall   # wipe and install from scratch
```

Verify:
```bash
forge --version
```

### Installer Options

| Flag | Description |
|------|-------------|
| `--prefix=/custom/path` | Custom install location |
| `--bin-dir=/custom/bin` | Directory for CLI symlinks |
| `--python=/path/to/python` | Python interpreter (must be 3.9+) |
| `--online` | Force install from source + PyPI |
| `--offline` | Force install from `offline_packages/` |
| `--update` | Keep the existing venv; only update the package and missing deps |
| `--reinstall` | Delete the previous install and install from scratch |
| `--no-system-packages` | Do not reuse libraries already installed on the system Python |
| `--dry-run` | Preview without installing |
| `--force` | Non-interactive reinstall of an existing prefix |
| `--yes`, `-y` | Non-interactive; update if already installed |
| `--skip-path` | Do not modify shell profile |
| `--uninstall` | Remove installation |

Environment variables `FORGE_INSTALL_PREFIX`, `FORGE_BIN_DIR`, and `FORGE_PYTHON`
are equivalent to `--prefix`, `--bin-dir`, and `--python`.

Non-interactive example (HPC batch / CI):
```bash
./install.sh --yes --offline --skip-path --prefix="$HOME/apps/forge"
export PATH="$HOME/apps/forge/bin:$PATH"
```

If the prompt shows `(base)` (conda), deactivate it first or pass system Python.
A conda pip index often fails with `No matching distribution found for setuptools`:
```bash
conda deactivate
sudo apt-get install -y python3-venv python3-pip
./install.sh --yes --python=/usr/bin/python3 --prefix="$HOME/apps/forge" --bin-dir="$HOME/apps/forge/bin"
```

If `pypi.org` times out, use a mirror:
```bash
./install.sh --yes --python=/usr/bin/python3 \
  --index-url=https://pypi.tuna.tsinghua.edu.cn/simple \
  --prefix="$HOME/apps/forge" --bin-dir="$HOME/apps/forge/bin"
```

### Root (System-Wide) Installation

```bash
sudo ./install.sh
```

Installs to `/opt/forge/` with symlinks in `/usr/local/bin/` and a PATH snippet
in `/etc/profile.d/forge.sh`.

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
make download-offline
# or: pip download -r offline_packages/requirements-offline.txt -d ./offline_packages/packaging_offline --only-binary=:all:
rsync -av . cluster:/path/to/forge/

# On the cluster:
cd /path/to/forge
./install.sh --offline --yes
```

The installer uses `offline_packages/packaging_offline/` (then `offline_packages/`)
when PyPI is unreachable or `--offline` is set. Bundled wheels currently target
CPython 3.9 x86_64; refresh them with `make download-offline` for other Python
versions or architectures.

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
