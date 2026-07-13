# Installation Guide

## Requirements

- **Python** >= 3.9
- **Linux** (x86_64 or aarch64) — full HPC features
- **macOS** — basic features only (no NUMA, no scheduler detection)
- **WIEN2k** installation with valid license

## Standard Installation

```bash
pip install forge
```

Verify:
```bash
forge --version
```

## From Source

```bash
git clone https://github.com/majidhaddad02/forge.git
cd forge
pip install -e ".[dev]"
```

## Air-Gapped HPC Installation

For clusters without internet access:

```bash
# On a machine with internet:
pip download forge -d ./offline_packages
rsync -av offline_packages/ cluster:/path/to/packages/

# On the cluster:
pip install --no-index --find-links /path/to/packages/ forge
```

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
