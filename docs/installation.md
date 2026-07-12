# Installation Guide

## Requirements

- **Python** >= 3.9
- **Linux** (x86_64 or aarch64) — full HPC features
- **macOS** — basic features only (no NUMA, no scheduler detection)
- **WIEN2k** installation with valid license

## Standard Installation

```bash
pip install wien2k_gen
```

Verify:
```bash
wien2k_gen --version
```

## From Source

```bash
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
pip install -e ".[dev]"
```

## Air-Gapped HPC Installation

For clusters without internet access:

```bash
# On a machine with internet:
pip download wien2k_gen -d ./offline_packages
rsync -av offline_packages/ cluster:/path/to/packages/

# On the cluster:
pip install --no-index --find-links /path/to/packages/ wien2k_gen
```

## Container Deployment

### Docker
```bash
docker build -t wien2k_gen .
docker run --rm -v $(pwd):/work wien2k_gen generate
```

### Singularity / Apptainer
```bash
<!-- TODO: add container registry URL when published -->
singularity build wien2k_gen.sif docker-daemon://wien2k_gen:latest
singularity exec --bind $(pwd):/work wien2k_gen.sif wien2k_gen generate
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
| `textual` | Interactive TUI (`wien2k_gen tui`) |
| `numpy` | Roofline model calculations |
| `psutil` | Enhanced process monitoring |

## Verifying Installation

```bash
wien2k_gen diagnostics
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
