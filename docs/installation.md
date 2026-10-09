# Installation Guide

FORGE ships as a source tree with `install.sh`, a Makefile, and Docker images. There is no published PyPI package named `forge`. Install from this repository, an offline wheel cache, or a container.

A valid WIEN2k license is still required to run generated jobs.

Tab-complete flags from `completions/forge.bash` (bash) and `completions/forge.zsh` (zsh). `install.sh` sources them via `$prefix/env.sh`.

---

## Requirements

- **Python** >= 3.9 with the `venv` module (`python3-venv` on Debian/Ubuntu)
- **Linux** (x86_64 or aarch64) for full HPC features
- **macOS** — basic features only (no NUMA, no scheduler detection)
- **WIEN2k** installation with a valid license

---

## Standard Installation (`install.sh`)

```bash
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
chmod +x install.sh
./install.sh
```

Creates a venv in `~/.local/opt/forge/`, installs this tree plus dependencies, and symlinks `forge`, `forge_sbatch`, and `forge_wizard` into `~/.local/bin/`. The script locates the repo from its own path, not from the current directory.

If NumPy, PyYAML, or Rich already exist on the selected Python, the installer reuses them via `--system-site-packages`. Pass `--no-system-packages` for a fully isolated venv.

If FORGE is already at the prefix, the installer asks **update** (keep venv) or **reinstall** (wipe). Non-interactive:

```bash
./install.sh --yes --update
./install.sh --yes --reinstall
```

Verify:

```bash
forge --version
```

---

## Installer Flags

Every flag below is from `install.sh --help`.

### `--prefix=PATH`

Install root. Default: `~/.local/opt/forge` (user) or `/opt/forge` (root).

```bash
./install.sh --prefix="$HOME/apps/forge"
./install.sh --prefix=/opt/forge
```

Equivalent: `FORGE_INSTALL_PREFIX`.

### `--bin-dir=PATH`

Directory for CLI symlinks. Default: `~/.local/bin` or `/usr/local/bin`.

```bash
./install.sh --bin-dir="$HOME/apps/forge/bin"
```

Equivalent: `FORGE_BIN_DIR`.

### `--python=PATH`

Interpreter (3.9+). Default: first of `python3.12` .. `python3.9`, then `python3`.

```bash
./install.sh --python=/usr/bin/python3
./install.sh --python=/usr/bin/python3.11 --yes
```

Equivalent: `FORGE_PYTHON`.

### `--index-url=URL`

pip index for **dependencies** (not a FORGE PyPI package). Default: `https://pypi.org/simple`.

```bash
./install.sh --yes --index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

Equivalent: `PIP_INDEX_URL`.

### `--online`

Force install from the current source tree, fetching wheels from the pip index.

```bash
./install.sh --online --yes
```

### `--offline`

Force install from `offline_packages/` (no network).

```bash
./install.sh --offline --yes
```

### `--update`

Keep an existing venv; upgrade the package and missing deps.

```bash
./install.sh --yes --update
```

### `--reinstall`

Delete the previous prefix tree and install from scratch.

```bash
./install.sh --yes --reinstall --prefix="$HOME/apps/forge"
```

### `--no-system-packages`

Do not reuse libraries already on the system Python.

```bash
./install.sh --yes --no-system-packages
```

### `--dry-run`

Preview without installing.

```bash
./install.sh --dry-run
./install.sh --dry-run --offline --prefix="$HOME/apps/forge"
```

### `--force`

Skip confirmation on overwrite / uninstall.

```bash
./install.sh --force --reinstall --yes
```

### `--yes`, `-y`

Non-interactive. Default action if already installed is update.

```bash
./install.sh --yes
./install.sh -y --offline --skip-path
```

### `--skip-path`

Do not modify `~/.bashrc` / `~/.zshrc`.

```bash
./install.sh --yes --skip-path --prefix="$HOME/apps/forge"
export PATH="$HOME/apps/forge/bin:$PATH"
```

### `--uninstall`

Remove the installation.

```bash
./install.sh --uninstall --force
```

### `--help`, `-h`

```bash
./install.sh --help
```

---

## Typical HPC / CI Install

```bash
./install.sh --yes --offline --skip-path --prefix="$HOME/apps/forge"
export PATH="$HOME/apps/forge/bin:$PATH"
```

If the prompt shows `(base)` (conda), deactivate first. A conda pip index often fails with `No matching distribution found for setuptools`:

```bash
conda deactivate
./install.sh --yes --python=/usr/bin/python3 --prefix="$HOME/apps/forge" --bin-dir="$HOME/apps/forge/bin"
```

If `pypi.org` times out, use `--index-url` (dependencies only):

```bash
./install.sh --yes --python=/usr/bin/python3 \
  --index-url=https://pypi.tuna.tsinghua.edu.cn/simple \
  --prefix="$HOME/apps/forge" --bin-dir="$HOME/apps/forge/bin"
```

---

## Root (System-Wide)

```bash
sudo ./install.sh
```

Installs to `/opt/forge/` with symlinks in `/usr/local/bin/` and a PATH snippet in `/etc/profile.d/forge.sh`.

---

## From Source (Developers)

```bash
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Makefile (same tree, still not PyPI):

```bash
make dev
make install
make minimal
```

| Target | What it does |
|--------|----------------|
| `make install` | Editable install of this repo |
| `make dev` | `.[dev]` plus pre-commit |
| `make minimal` | Core deps only (no Textual TUI) |

---

## Air-Gapped HPC

```bash
make download-offline
rsync -av . cluster:/path/to/wien2k_gen/

cd /path/to/wien2k_gen
./install.sh --offline --yes
```

Bundled wheels currently target CPython 3.9 x86_64. Refresh with `make download-offline` for other Python versions or architectures.

---

## Container Deployment

Runtime image includes Python, OpenMPI, gfortran, and `forge[hpc]`. Dev image adds pytest/ruff/mypy. WIEN2k itself is **not** baked in (license).

```bash
make docker
make docker-dev
docker run --rm -v "$PWD":/work -w /work forge:0.1.0 forge generate
docker compose run --rm forge --help
make docker-test
make docker-shell
```

Singularity / Apptainer from a local image:

```bash
singularity build forge.sif docker-daemon://forge:latest
singularity exec --bind $(pwd):/work forge.sif forge generate
```

Or `make singularity` (`apptainer build --force forge.sif Singularity.def`).

---

## Shell Tab Completion

Files in `completions/`:

| File | Shell | Command |
|------|-------|---------|
| `completions/forge.bash` | bash | `forge` |
| `completions/forge.zsh` | zsh | `forge` |
| `completions/forge_sbatch.bash` | bash | `forge_sbatch` |
| `completions/forge_sbatch.zsh` | zsh | `forge_sbatch` |
| `completions/forge_wizard.bash` | bash | `forge_wizard` |
| `completions/forge_wizard.zsh` | zsh | `forge_wizard` |

`install.sh` writes `$prefix/env.sh` (PATH + completion) and appends a marked block to `~/.bashrc` and `~/.zshrc`. Completions do not need the `bash-completion` package.

```bash
forge <Tab>
forge generate --<Tab>
forge_sbatch <Tab>
```

Same session: `source ~/.bashrc` or `source ~/.zshrc`. If Tab still does nothing, `source ~/.local/opt/forge/env.sh`.

Dev fallback without the installer:

```bash
source completions/forge.bash
make install-completions
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WIENROOT` | Path to WIEN2k | auto-detected |
| `SCRATCH` | Scratch directory | `/tmp` |
| `TMPDIR` | Temporary files | `$SCRATCH` |
| `LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR | INFO |
| `NO_COLOR` | Disable colored output | unset |
| `FORGE_INSTALL_PREFIX` | Same as `--prefix` | unset |
| `FORGE_BIN_DIR` | Same as `--bin-dir` | unset |
| `FORGE_PYTHON` | Same as `--python` | unset |
| `PIP_INDEX_URL` | Same as `--index-url` | PyPI simple |

---

## Optional Dependencies

| Package | Needed For |
|---------|------------|
| `rich` | Colored CLI (installed by default) |
| `textual` | `forge tui` |
| `numpy` | Roofline calculations |
| `psutil` | Process monitoring |

---

## Verifying Installation

```bash
forge diagnostics
forge diagnostics --full
forge diagnostics --export report.json
```

Shows CPU, physical vs logical cores, NUMA, scheduler, MPI, WIEN2k path/version, scratch filesystem, and interconnect.
