#!/usr/bin/env bash
# ==============================================================================
# FORGE Installer (v0.1.0) – Production-Grade HPC Setup Script
# Supports root/user installation, online/offline modes, safe cleanup,
# automatic verification, and seamless integration with pyproject.toml
# ==============================================================================
set -euo pipefail

# Configuration
APP_NAME="forge"
APP_VERSION="0.1.0"
OFFLINE_DIR_NAME="offline_packages"
BINARIES=("forge" "forge_sbatch" "forge_wizard")
MARKER_BEGIN="# >>> forge begin >>>"
MARKER_END="# <<< forge end <<<"
INSTALL_MARKER=".forge-installed"

# Resolve repository root from this script's location (not the caller's cwd).
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [[ -L "${SCRIPT_PATH}" ]]; do
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  SCRIPT_PATH="$(readlink "${SCRIPT_PATH}")"
  [[ "${SCRIPT_PATH}" != /* ]] && SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_PATH}"
done
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
OFFLINE_DIR="${REPO_ROOT}/${OFFLINE_DIR_NAME}"

# Colors & Logging (disabled when not a TTY or NO_COLOR is set)
if [[ -n "${NO_COLOR:-}" || ! -t 1 ]]; then
  RED=""; GREEN=""; YELLOW=""; CYAN=""; NC=""
else
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
fi
log()     { echo -e "${CYAN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

is_tty() { [[ -t 0 && -t 1 ]]; }

python_is_conda() {
  local py="$1"
  local prefix
  prefix="$("${py}" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
  [[ -n "${prefix}" && ( -n "${CONDA_PREFIX:-}" && "${prefix}" == "${CONDA_PREFIX}"* ) ]] && return 0
  [[ "${prefix}" == *"/miniconda"* || "${prefix}" == *"/anaconda"* || "${prefix}" == *"/miniforge"* || "${prefix}" == *"/mambaforge"* ]] && return 0
  [[ "${prefix}" == *"/envs/"* ]] && return 0
  return 1
}

DEFAULT_PIP_INDEX="https://pypi.org/simple"
PIP_MIRRORS=(
  "https://pypi.org/simple"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple"
  "https://pypi.mirrors.ustc.edu.cn/simple"
)

sanitize_pip_env() {
  # Conda and site pip.conf often set an empty/private index, which yields:
  # "Could not find a version that satisfies the requirement setuptools (from versions: none)"
  unset PIP_NO_INDEX PIP_FIND_LINKS || true
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
  export PIP_INDEX_URL="${PIP_INDEX_URL:-${DEFAULT_PIP_INDEX}}"
}

pip_indexes_to_try() {
  local seen="|"
  local u
  for u in "${PIP_INDEX_URL}" "${DEFAULT_PIP_INDEX}" "${PIP_MIRRORS[@]}"; do
    [[ -n "${u}" ]] || continue
    [[ "${seen}" == *"|${u}|"* ]] && continue
    seen="${seen}${u}|"
    printf '%s\n' "${u}"
  done
}

pip_install_online() {
  local idx
  local last_err=1
  sanitize_pip_env
  while IFS= read -r idx; do
    log "pip index: ${idx}"
    if "${PYTHON_VENV}" -m pip install --index-url "${idx}" --timeout 60 "$@"; then
      return 0
    fi
    last_err=$?
    warn "pip failed with index ${idx}"
  done < <(pip_indexes_to_try)
  return "${last_err}"
}

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --prefix=PATH     Install root (default: ~/.local/opt/forge or /opt/forge for root)
  --bin-dir=PATH    Symlink directory for CLI binaries (default: ~/.local/bin or /usr/local/bin)
  --python=PATH     Python interpreter (default: python3.12 .. python3.9, then python3)
  --index-url=URL   pip index (default: https://pypi.org/simple)
  --online          Force online install from PyPI / current source tree
  --offline         Force offline install from ${OFFLINE_DIR_NAME}/
  --update          Keep an existing install and only update the package/deps
  --reinstall       Wipe an existing install and install from scratch
  --no-system-packages  Do not reuse packages already on the system Python
  --dry-run         Preview without installing
  --force           Skip confirmation on overwrite / uninstall
  --yes, -y         Non-interactive; default is update if already installed
  --skip-path       Do not modify shell profile
  --uninstall       Remove installation
  --help, -h        Show this message

Environment variables (overridden by CLI flags):
  FORGE_INSTALL_PREFIX   Same as --prefix
  FORGE_BIN_DIR          Same as --bin-dir
  FORGE_PYTHON           Same as --python
  PIP_INDEX_URL          Same as --index-url
  NO_COLOR               Disable colored output
EOF
}

# CLI Flags
UNINSTALL=false
DRY_RUN=false
FORCE=false
YES=false
SKIP_PATH=false
NO_SYSTEM_PACKAGES=false
MODE=""   # "", online, offline
EXISTING_ACTION=""  # "", update, reinstall
INSTALL_MODE="fresh"  # fresh, update, reinstall
PREFIX_OVERRIDE=""
BIN_DIR_OVERRIDE=""
PYTHON_OVERRIDE="${FORGE_PYTHON:-}"
INDEX_URL_OVERRIDE=""

require_value() {
  local flag="$1"
  local value="${2:-}"
  [[ -n "${value}" && "${value}" != --* ]] || error "${flag} requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uninstall) UNINSTALL=true; shift ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --force)     FORCE=true; shift ;;
    --yes|-y)    YES=true; shift ;;
    --skip-path) SKIP_PATH=true; shift ;;
    --online)    MODE="online"; shift ;;
    --offline)   MODE="offline"; shift ;;
    --update)    EXISTING_ACTION="update"; shift ;;
    --reinstall) EXISTING_ACTION="reinstall"; shift ;;
    --no-system-packages) NO_SYSTEM_PACKAGES=true; shift ;;
    --prefix=*)  PREFIX_OVERRIDE="${1#*=}"; shift ;;
    --prefix)
      require_value "$1" "${2:-}"
      PREFIX_OVERRIDE="$2"; shift 2
      ;;
    --bin-dir=*) BIN_DIR_OVERRIDE="${1#*=}"; shift ;;
    --bin-dir)
      require_value "$1" "${2:-}"
      BIN_DIR_OVERRIDE="$2"; shift 2
      ;;
    --python=*)  PYTHON_OVERRIDE="${1#*=}"; shift ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_OVERRIDE="$2"; shift 2
      ;;
    --index-url=*) INDEX_URL_OVERRIDE="${1#*=}"; shift ;;
    --index-url)
      require_value "$1" "${2:-}"
      INDEX_URL_OVERRIDE="$2"; shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      error "Unknown option: $1 (try --help)"
      ;;
  esac
done

if [[ -n "${INDEX_URL_OVERRIDE}" ]]; then
  PIP_INDEX_URL="${INDEX_URL_OVERRIDE}"
fi
: "${PIP_INDEX_URL:=${DEFAULT_PIP_INDEX}}"

if $UNINSTALL && [[ "${MODE}" == "online" || "${MODE}" == "offline" ]]; then
  warn "--online/--offline ignored during uninstall."
fi

# ==============================================================================
# 1. Determine Installation Prefix, PATH dir, and profile
# ==============================================================================
: "${FORGE_INSTALL_PREFIX:=}"
: "${FORGE_BIN_DIR:=}"

if [[ -n "${PREFIX_OVERRIDE}" ]]; then
  INSTALL_PREFIX="${PREFIX_OVERRIDE}"
elif [[ -n "${FORGE_INSTALL_PREFIX}" ]]; then
  INSTALL_PREFIX="${FORGE_INSTALL_PREFIX}"
elif [[ "$(id -u)" -eq 0 ]]; then
  INSTALL_PREFIX="/opt/${APP_NAME}"
else
  INSTALL_PREFIX="${HOME}/.local/opt/${APP_NAME}"
fi

if [[ -n "${BIN_DIR_OVERRIDE}" ]]; then
  BIN_LINK_DIR="${BIN_DIR_OVERRIDE}"
elif [[ -n "${FORGE_BIN_DIR}" ]]; then
  BIN_LINK_DIR="${FORGE_BIN_DIR}"
elif [[ -n "${PREFIX_OVERRIDE}" || -n "${FORGE_INSTALL_PREFIX}" ]]; then
  BIN_LINK_DIR="${INSTALL_PREFIX}/bin"
elif [[ "$(id -u)" -eq 0 ]]; then
  BIN_LINK_DIR="/usr/local/bin"
else
  BIN_LINK_DIR="${HOME}/.local/bin"
fi

to_abs() {
  local p="$1"
  [[ "${p}" == ~* ]] && p="${p/#\~/${HOME}}"
  if [[ "${p}" == /* ]]; then
    printf '%s\n' "${p}"
  else
    printf '%s\n' "${PWD}/${p}"
  fi
}

INSTALL_PREFIX="$(to_abs "${INSTALL_PREFIX}")"
BIN_LINK_DIR="$(to_abs "${BIN_LINK_DIR}")"

VENV_DIR="${INSTALL_PREFIX}/venv"
PIP="${VENV_DIR}/bin/pip"
PYTHON_VENV="${VENV_DIR}/bin/python"

detect_profile_file() {
  if [[ "$(id -u)" -eq 0 ]]; then
    echo "/etc/profile.d/${APP_NAME}.sh"
    return
  fi
  local shell_name
  shell_name="$(basename "${SHELL:-bash}")"
  case "${shell_name}" in
    zsh)
      echo "${HOME}/.zshrc"
      ;;
    bash)
      if [[ -f "${HOME}/.bashrc" ]]; then
        echo "${HOME}/.bashrc"
      elif [[ -f "${HOME}/.bash_profile" ]]; then
        echo "${HOME}/.bash_profile"
      else
        echo "${HOME}/.bashrc"
      fi
      ;;
    fish)
      echo ""
      ;;
    *)
      if [[ -f "${HOME}/.profile" ]]; then
        echo "${HOME}/.profile"
      else
        echo "${HOME}/.bashrc"
      fi
      ;;
  esac
}

PROFILE_FILE="$(detect_profile_file)"

is_unsafe_prefix() {
  local p="$1"
  local home="${HOME}"
  case "${p}" in
    /|/usr|/usr/local|/usr/local/bin|/usr/bin|/bin|/sbin|/etc|/opt|/var|/home|"${home}"|"${home}/.local"|"${home}/.local/bin"|"${home}/bin")
      return 0
      ;;
  esac
  return 1
}

is_forge_install() {
  local p="$1"
  [[ -f "${p}/${INSTALL_MARKER}" ]] && return 0
  [[ -x "${p}/venv/bin/forge" ]] && return 0
  [[ -x "${p}/venv/bin/python" ]] && "${p}/venv/bin/python" -c "import forge" >/dev/null 2>&1 && return 0
  return 1
}

confirm() {
  local prompt="$1"
  local default="${2:-n}"
  local ans
  if $FORCE || $YES; then
    return 0
  fi
  if ! is_tty; then
    error "Refusing to prompt in non-interactive mode. Re-run with --force or --yes. (${prompt})"
  fi
  if [[ "${default}" == "y" ]]; then
    read -rp "${prompt} [Y/n]: " ans || true
    [[ -z "${ans}" || "${ans}" =~ ^[Yy]$ ]]
  else
    read -rp "${prompt} [y/N]: " ans || true
    [[ "${ans}" =~ ^[Yy]$ ]]
  fi
}

choose_existing_action() {
  if [[ -n "${EXISTING_ACTION}" ]]; then
    printf '%s\n' "${EXISTING_ACTION}"
    return
  fi
  if $FORCE; then
    printf '%s\n' "reinstall"
    return
  fi
  if $YES; then
    printf '%s\n' "update"
    return
  fi
  if ! is_tty; then
    error "Existing installation found at ${INSTALL_PREFIX}. Re-run with --update, --reinstall, or --yes."
  fi
  echo
  warn "FORGE is already installed at ${INSTALL_PREFIX}"
  echo "  [1] Update     keep venv; install only missing/outdated packages (default)"
  echo "  [2] Reinstall  delete the previous install and install from scratch"
  echo "  [3] Cancel"
  local ans
  read -rp "Choose [1/2/3]: " ans || true
  case "${ans}" in
    2|r|R|reinstall) printf '%s\n' "reinstall" ;;
    3|n|N|c|C|q|Q)   printf '%s\n' "cancel" ;;
    *)               printf '%s\n' "update" ;;
  esac
}

list_satisfied_deps() {
  local py="$1"
  "${py}" - <<'PY' 2>/dev/null || true
import importlib.metadata as md

need = {
    "rich": "13.0",
    "textual": "0.40",
    "PyYAML": "6.0",
    "typing_extensions": "4.5",
    "numpy": "1.24",
    "psutil": "5.9",
    "filelock": "3.12",
    "packaging": "23.0",
}

def ver_tuple(s):
    parts = []
    for p in s.split("."):
        n = ""
        for ch in p:
            if ch.isdigit():
                n += ch
            else:
                break
        parts.append(int(n or 0))
    return tuple(parts)

found = []
missing = []
for name, minv in need.items():
    try:
        cur = md.version(name)
    except Exception:
        missing.append(name)
        continue
    if ver_tuple(cur) >= ver_tuple(minv):
        found.append(f"{name}=={cur}")
    else:
        missing.append(f"{name}=={cur}(<{minv})")
print("FOUND:" + ",".join(found))
print("MISSING:" + ",".join(missing))
PY
}

USE_SYSTEM_SITE=false
SATISFIED_DEPS=""

inspect_host_packages() {
  $NO_SYSTEM_PACKAGES && return 0
  local report found missing
  report="$(list_satisfied_deps "${PYTHON_BIN}")"
  found="$(printf '%s\n' "${report}" | awk -F: '/^FOUND:/{print substr($0,7)}')"
  missing="$(printf '%s\n' "${report}" | awk -F: '/^MISSING:/{print substr($0,9)}')"
  if [[ -n "${found}" ]]; then
    SATISFIED_DEPS="${found}"
    USE_SYSTEM_SITE=true
    log "Reusing already-installed packages: ${found//,/, }"
  else
    log "No reusable host packages found for core dependencies."
  fi
  if [[ -n "${missing}" ]]; then
    log "Will install/upgrade: ${missing//,/, }"
  fi
}

# ==============================================================================
# Helper: Python interpreter, wheels, internet, profile block
# ==============================================================================
detect_python() {
  if [[ -n "${PYTHON_OVERRIDE}" ]]; then
    if ! command -v "${PYTHON_OVERRIDE}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_OVERRIDE}" ]]; then
      error "Python interpreter not found: ${PYTHON_OVERRIDE}"
    fi
    if ! "${PYTHON_OVERRIDE}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      error "Python >= 3.9 is required (got ${PYTHON_OVERRIDE})."
    fi
    if python_is_conda "${PYTHON_OVERRIDE}"; then
      warn "Using conda Python (${PYTHON_OVERRIDE}). Prefer system Python if pip index errors appear."
    fi
    echo "${PYTHON_OVERRIDE}"
    return
  fi
  local c resolved
  # Prefer distro Python; conda (base) often ships a broken pip/ensurepip index.
  for c in /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3.9 /usr/bin/python3 \
           python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "${c}" >/dev/null 2>&1 || [[ -x "${c}" ]]; then
      resolved="$(command -v "${c}" 2>/dev/null || echo "${c}")"
      if python_is_conda "${resolved}"; then
        continue
      fi
      if "${resolved}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        echo "${resolved}"
        return
      fi
    fi
  done
  for c in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "${c}" >/dev/null 2>&1; then
      resolved="$(command -v "${c}")"
      if "${resolved}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        warn "Only conda/non-system Python found: ${resolved}"
        echo "${resolved}"
        return
      fi
    fi
  done
  return 1
}

find_wheel_dir() {
  local d
  for d in \
    "${OFFLINE_DIR}/packaging_offline" \
    "${OFFLINE_DIR}" \
    "${REPO_ROOT}/packaging_offline"
  do
    if [[ -d "${d}" ]] && ls -A "${d}"/*.whl >/dev/null 2>&1; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

check_internet() {
  # Prefer HTTPS; ICMP ping is often blocked on HPC login nodes.
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --head -m 3 https://pypi.org >/dev/null 2>&1 && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=3)" >/dev/null 2>&1 && return 0
  fi
  if command -v ping >/dev/null 2>&1; then
    ping -c 1 -W 2 pypi.org >/dev/null 2>&1 && return 0
  fi
  return 1
}

remove_profile_block() {
  local f="$1"
  [[ -n "${f}" && -f "${f}" ]] || return 0
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/forge-profile.XXXXXX")"
  # New installer uses begin/end markers. Also drop the two-line block from
  # older install.sh versions: "# forge vX.Y.Z Environment" + export PATH=...
  awk -v b="${MARKER_BEGIN}" -v e="${MARKER_END}" -v app="${APP_NAME}" '
    $0 == b {skip=1; next}
    $0 == e {skip=0; next}
    skip {next}
    $0 ~ ("^# " app " v[0-9].* Environment") {old=1; next}
    old && $0 ~ /^export PATH=/ {old=0; next}
    old {old=0}
    {print}
  ' "${f}" > "${tmp}"
  cat "${tmp}" > "${f}"
}

write_profile_block() {
  local f="$1"
  local path_dir="$2"
  [[ -n "${f}" ]] || return 0
  mkdir -p "$(dirname "${f}")"
  touch "${f}"
  remove_profile_block "${f}"
  {
    echo ""
    echo "${MARKER_BEGIN}"
    echo "# ${APP_NAME} v${APP_VERSION} environment"
    echo "export PATH=\"${path_dir}:\${PATH}\""
    echo "${MARKER_END}"
  } >> "${f}"
}

print_plan() {
  log "Install prefix: ${INSTALL_PREFIX}"
  log "Bin directory:  ${BIN_LINK_DIR}"
  log "Venv directory: ${VENV_DIR}"
  log "Repo root:      ${REPO_ROOT}"
  if [[ -n "${PROFILE_FILE}" ]]; then
    log "Profile file:   ${PROFILE_FILE}"
  else
    log "Profile file:   (skipped; unsupported shell)"
  fi
}

# ==============================================================================
# 2. Uninstall Mode
# ==============================================================================
do_uninstall() {
  print_plan
  log "Uninstalling ${APP_NAME} from ${INSTALL_PREFIX}..."

  if [[ -d "${INSTALL_PREFIX}" ]]; then
    if is_unsafe_prefix "${INSTALL_PREFIX}"; then
      error "Refusing to remove unsafe path: ${INSTALL_PREFIX}"
    fi
    if ! is_forge_install "${INSTALL_PREFIX}"; then
      if ! confirm "Path ${INSTALL_PREFIX} does not look like a FORGE install. Remove it anyway?"; then
        log "Uninstall cancelled."
        exit 0
      fi
    elif ! $DRY_RUN; then
      confirm "Remove ${INSTALL_PREFIX}?" || { log "Uninstall cancelled."; exit 0; }
    fi
    if $DRY_RUN; then
      log "DRY-RUN: would remove ${INSTALL_PREFIX}"
    else
      # The install tree is owned by this installer; recursive removal is the uninstall path.
      rm -rf "${INSTALL_PREFIX}"
    fi
  else
    warn "Install prefix not found: ${INSTALL_PREFIX}"
  fi

  local bin
  for bin in "${BINARIES[@]}"; do
    local link="${BIN_LINK_DIR}/${bin}"
    if [[ -L "${link}" || -f "${link}" ]]; then
      if $DRY_RUN; then
        log "DRY-RUN: would remove ${link}"
      else
        rm -f "${link}"
      fi
    fi
  done

  if [[ "$(id -u)" -eq 0 && -f "/etc/profile.d/${APP_NAME}.sh" ]]; then
    if $DRY_RUN; then
      log "DRY-RUN: would remove /etc/profile.d/${APP_NAME}.sh"
    else
      rm -f "/etc/profile.d/${APP_NAME}.sh"
    fi
  elif [[ -n "${PROFILE_FILE}" && -f "${PROFILE_FILE}" ]]; then
    if $DRY_RUN; then
      log "DRY-RUN: would remove PATH block from ${PROFILE_FILE}"
    else
      remove_profile_block "${PROFILE_FILE}"
    fi
  fi

  success "${APP_NAME} uninstalled."
  if [[ -n "${PROFILE_FILE}" ]]; then
    log "Restart the shell or run: source ${PROFILE_FILE}"
  fi
}

if $UNINSTALL; then
  do_uninstall
  exit 0
fi

# ==============================================================================
# 3. Pre-flight checks
# ==============================================================================
[[ -f "${REPO_ROOT}/pyproject.toml" ]] || error "pyproject.toml not found in ${REPO_ROOT}. Run this installer from a FORGE checkout."

if is_unsafe_prefix "${INSTALL_PREFIX}"; then
  error "Refusing unsafe --prefix: ${INSTALL_PREFIX}"
fi

PYTHON_BIN="$(detect_python)" || error "Python >= 3.9 is required. Install python3 or pass --python=/path/to/python."
PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "Using Python ${PY_VERSION} (${PYTHON_BIN})"

if ! "${PYTHON_BIN}" -c "import venv, ensurepip" >/dev/null 2>&1; then
  if $DRY_RUN; then
    warn "Python venv/ensurepip is missing. On Debian/Ubuntu install python3-venv."
  else
    error "Python venv/ensurepip is missing. On Debian/Ubuntu install python3-venv."
  fi
fi

print_plan

if $DRY_RUN; then
  log "Dry-run: no files will be changed."
fi

# ==============================================================================
# 4. Existing Installation Check
# ==============================================================================
if [[ -d "${INSTALL_PREFIX}" ]]; then
  if is_forge_install "${INSTALL_PREFIX}" || [[ -d "${VENV_DIR}" ]]; then
    INSTALL_MODE="$(choose_existing_action)"
    if [[ "${INSTALL_MODE}" == "cancel" ]]; then
      log "Installation cancelled."
      exit 0
    fi
    if [[ "${INSTALL_MODE}" == "reinstall" ]]; then
      if $DRY_RUN; then
        log "DRY-RUN: would wipe ${INSTALL_PREFIX} and reinstall"
      else
        log "Removing previous installation at ${INSTALL_PREFIX}..."
        if is_unsafe_prefix "${INSTALL_PREFIX}"; then
          error "Refusing to remove unsafe path: ${INSTALL_PREFIX}"
        fi
        rm -rf "${INSTALL_PREFIX}"
        for bin in "${BINARIES[@]}"; do
          rm -f "${BIN_LINK_DIR}/${bin}" 2>/dev/null || true
        done
      fi
    else
      log "Updating existing installation (venv kept)."
    fi
  elif $FORCE || $YES; then
    warn "Overwriting non-FORGE path ${INSTALL_PREFIX} because --force/--yes was set."
    INSTALL_MODE="reinstall"
    if ! $DRY_RUN; then
      rm -rf "${INSTALL_PREFIX}"
    fi
  else
    error "Refusing to overwrite ${INSTALL_PREFIX} (not a FORGE install). Re-run with --force after confirming the path."
  fi
fi

inspect_host_packages

# ==============================================================================
# 5. Internet / offline decision
# ==============================================================================
WHEEL_DIR=""
WHEEL_DIR="$(find_wheel_dir || true)"

USE_ONLINE=false
case "${MODE}" in
  online)  USE_ONLINE=true ;;
  offline) USE_ONLINE=false ;;
  *)
    if check_internet; then
      log "Internet detected."
      if $DRY_RUN || $YES || ! is_tty; then
        USE_ONLINE=true
      else
        if confirm "Download packages matching this Python version?" "y"; then
          USE_ONLINE=true
        fi
      fi
    else
      log "No internet connection (or PyPI unreachable)."
      USE_ONLINE=false
    fi
    ;;
esac

if ! $USE_ONLINE; then
  if [[ -z "${WHEEL_DIR}" ]]; then
    error "Offline directory '${OFFLINE_DIR}' has no wheel files. Place wheels in ${OFFLINE_DIR_NAME}/packaging_offline/ or re-run with network / --online."
  fi
  log "Offline wheel directory: ${WHEEL_DIR}"
  PY_TAG="$("${PYTHON_BIN}" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
  ARCH="$("${PYTHON_BIN}" -c 'import platform; print(platform.machine())')"
  if ! ls "${WHEEL_DIR}"/numpy-*-"${PY_TAG}"-*.whl >/dev/null 2>&1 \
     && ls "${WHEEL_DIR}"/numpy-*-cp*.whl >/dev/null 2>&1; then
    warn "Offline numpy wheels may not match Python ${PY_VERSION} (${PY_TAG}) / ${ARCH}."
    warn "Bundled wheels target CPython 3.9 x86_64. Use --python=python3.9 or --online, or refresh wheels with: make download-offline"
  fi
fi

if $DRY_RUN; then
  log "DRY-RUN mode: ${INSTALL_MODE}"
  if $USE_SYSTEM_SITE; then
    log "DRY-RUN: venv --system-site-packages (reuse host packages)"
  fi
  if $USE_ONLINE; then
    if [[ "${INSTALL_MODE}" == "update" && -x "${PYTHON_VENV}" ]]; then
      log "DRY-RUN: pip install --upgrade '${REPO_ROOT}'"
    else
      log "DRY-RUN: ${PYTHON_BIN} -m venv ${VENV_DIR}"
      log "DRY-RUN: pip install --upgrade pip setuptools wheel"
      log "DRY-RUN: pip install '${REPO_ROOT}'"
    fi
  else
    log "DRY-RUN: pip install --no-index --no-build-isolation --upgrade --find-links='${WHEEL_DIR}' '${REPO_ROOT}'"
  fi
  log "DRY-RUN: symlink binaries into ${BIN_LINK_DIR}"
  if ! $SKIP_PATH && [[ -n "${PROFILE_FILE}" ]]; then
    log "DRY-RUN: add PATH to ${PROFILE_FILE}"
  fi
  success "Dry-run completed."
  exit 0
fi

# ==============================================================================
# 6. Installation (uses venv, not --prefix)
# ==============================================================================
ensure_venv() {
  mkdir -p "${INSTALL_PREFIX}"
  if [[ -x "${PYTHON_VENV}" && "${INSTALL_MODE}" == "update" ]]; then
    log "Reusing existing virtual environment at ${VENV_DIR}."
  else
    log "Creating virtual environment at ${VENV_DIR}..."
    local venv_flags=()
    $USE_SYSTEM_SITE && venv_flags+=(--system-site-packages)
    # --without-pip avoids contacting PyPI during venv creation (timeouts on some networks).
    if ! "${PYTHON_BIN}" -m venv --without-pip "${venv_flags[@]}" "${VENV_DIR}" 2>/dev/null; then
      if ! "${PYTHON_BIN}" -m venv "${venv_flags[@]}" "${VENV_DIR}"; then
        error "Failed to create venv with ${PYTHON_BIN}. On Ubuntu: sudo apt-get install -y python3-venv python3-pip"
      fi
    fi
  fi
  [[ -x "${PYTHON_VENV}" ]] || error "venv created but python missing at ${VENV_DIR}"
  if [[ ! -x "${PIP}" ]]; then
    warn "venv pip missing; bootstrapping with ensurepip..."
    "${PYTHON_VENV}" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi
  [[ -x "${PIP}" ]] || error "venv created but pip missing at ${VENV_DIR}. On Ubuntu: sudo apt-get install -y python3-venv"
  if $USE_SYSTEM_SITE; then
    cfg="${VENV_DIR}/pyvenv.cfg"
    if [[ -f "${cfg}" ]] && ! grep -qiE '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${cfg}"; then
      if grep -qiE '^include-system-site-packages' "${cfg}"; then
        sed -i 's/^include-system-site-packages.*/include-system-site-packages = true/' "${cfg}"
      else
        echo "include-system-site-packages = true" >> "${cfg}"
      fi
      log "Enabled system-site-packages so host libraries are reused."
    fi
  fi
}

ensure_venv

if $USE_ONLINE; then
  log "Installing from source + pip index..."
  sanitize_pip_env
  if ! pip_install_online --upgrade pip setuptools wheel; then
    warn "Could not upgrade pip/setuptools from the network; continuing with venv pip."
  fi
  if ! pip_install_online --upgrade "${REPO_ROOT}"; then
    if [[ -n "${WHEEL_DIR:-}" ]]; then
      warn "Online install failed; falling back to offline wheels in ${WHEEL_DIR}."
      USE_ONLINE=false
    else
      error "Online install failed (PyPI unreachable). Re-run with --offline, or pass --index-url=https://pypi.tuna.tsinghua.edu.cn/simple"
    fi
  fi
fi

if ! $USE_ONLINE; then
  log "Installing from offline packages..."
  # Bundled ensurepip already provides pip/setuptools; do not hit PyPI.
  # --no-build-isolation avoids downloading setuptools to build the local sdist.
  req_file="${OFFLINE_DIR}/requirements-offline.txt"
  if [[ "${INSTALL_MODE}" != "update" ]]; then
    if [[ -f "${req_file}" ]] && "${PIP}" install --no-index --find-links="${WHEEL_DIR}" -r "${req_file}"; then
      :
    else
      warn "Full offline requirements install failed; installing available compatible wheels."
      if ! "${PIP}" install --no-index --find-links="${WHEEL_DIR}" \
        rich textual pyyaml typing-extensions numpy psutil \
        markdown-it-py pygments platformdirs mdurl linkify-it-py uc-micro-py; then
        error "Offline dependency install failed for Python ${PY_VERSION}. Bundled wheels target CPython 3.9 x86_64."
      fi
    fi
  else
    log "Update mode: skipping reinstall of already-present offline wheels."
  fi
  "${PIP}" install --no-index --no-build-isolation --no-deps --upgrade --find-links="${WHEEL_DIR}" "${REPO_ROOT}"
fi

printf '%s\n' "version=${APP_VERSION}" "prefix=${INSTALL_PREFIX}" > "${INSTALL_PREFIX}/${INSTALL_MARKER}"
success "Package installed."

# ==============================================================================
# 7. Symlinks & Environment Setup
# ==============================================================================
log "Creating symlinks in ${BIN_LINK_DIR}..."
mkdir -p "${BIN_LINK_DIR}"
for bin in "${BINARIES[@]}"; do
  target="${VENV_DIR}/bin/${bin}"
  if [[ -f "${target}" ]]; then
    ln -sf "${target}" "${BIN_LINK_DIR}/${bin}"
  else
    warn "Missing binary: ${bin}"
  fi
done

if $SKIP_PATH; then
  log "Skipping PATH profile modification (--skip-path)."
elif [[ -z "${PROFILE_FILE}" ]]; then
  warn "Unsupported shell (${SHELL:-unknown}). Add this to PATH manually:"
  warn "  export PATH=\"${BIN_LINK_DIR}:\$PATH\""
elif [[ "$(id -u)" -eq 0 ]]; then
  mkdir -p "$(dirname "${PROFILE_FILE}")"
  cat > "${PROFILE_FILE}" <<EOF
${MARKER_BEGIN}
# ${APP_NAME} v${APP_VERSION} environment
export PATH="${BIN_LINK_DIR}:\${PATH}"
${MARKER_END}
EOF
  log "Wrote ${PROFILE_FILE}"
else
  case ":${PATH}:" in
    *":${BIN_LINK_DIR}:"*)
      log "PATH already contains ${BIN_LINK_DIR}."
      ;;
  esac
  write_profile_block "${PROFILE_FILE}" "${BIN_LINK_DIR}"
  log "Added PATH to ${PROFILE_FILE}. Run: source ${PROFILE_FILE}"
fi

# Optional shell completions (best-effort, never fail the install).
COMPLETIONS_SRC="${REPO_ROOT}/completions"
if [[ -d "${COMPLETIONS_SRC}" ]] && [[ "$(id -u)" -ne 0 ]]; then
  mkdir -p "${HOME}/.local/share/bash-completion/completions" "${HOME}/.local/share/zsh/site-functions" 2>/dev/null || true
  cp "${COMPLETIONS_SRC}/forge.bash" "${HOME}/.local/share/bash-completion/completions/forge" 2>/dev/null || true
  cp "${COMPLETIONS_SRC}/forge_sbatch.bash" "${HOME}/.local/share/bash-completion/completions/forge_sbatch" 2>/dev/null || true
  cp "${COMPLETIONS_SRC}/forge_wizard.bash" "${HOME}/.local/share/bash-completion/completions/forge_wizard" 2>/dev/null || true
  cp "${COMPLETIONS_SRC}/forge.zsh" "${HOME}/.local/share/zsh/site-functions/_forge" 2>/dev/null || true
  cp "${COMPLETIONS_SRC}/forge_sbatch.zsh" "${HOME}/.local/share/zsh/site-functions/_forge_sbatch" 2>/dev/null || true
  cp "${COMPLETIONS_SRC}/forge_wizard.zsh" "${HOME}/.local/share/zsh/site-functions/_forge_wizard" 2>/dev/null || true
fi

# ==============================================================================
# 8. Post-Install Verification
# ==============================================================================
log "Running verification..."
BIN_PATH="${BIN_LINK_DIR}/forge"

if [[ ! -x "${BIN_PATH}" ]]; then
  error "Binary not found at ${BIN_PATH}. Installation may have failed."
fi

VER_OUT="$("${PYTHON_VENV}" -c "import forge; print(forge.__version__)" 2>/dev/null || true)"
if [[ "${VER_OUT}" == *"${APP_VERSION}"* ]]; then
  success "Version check passed: ${VER_OUT}"
else
  CLI_VER="$("${BIN_PATH}" --version 2>/dev/null || true)"
  if [[ "${CLI_VER}" == *"${APP_VERSION}"* ]]; then
    success "Version check passed: ${CLI_VER}"
  else
    warn "Version mismatch. module=${VER_OUT:-empty} cli=${CLI_VER:-empty}"
  fi
fi

if "${PYTHON_VENV}" -c "import ${APP_NAME}" >/dev/null 2>&1; then
  success "Python module import passed."
else
  error "Module import failed in ${PYTHON_VENV}."
fi

success "Installation completed successfully."
log "Next steps:"
if $SKIP_PATH || [[ -z "${PROFILE_FILE}" ]]; then
  log "  1. export PATH=\"${BIN_LINK_DIR}:\$PATH\""
else
  log "  1. source ${PROFILE_FILE}"
fi
log "  2. ${APP_NAME} --help"
log "  3. ${APP_NAME} tui"
