#!/usr/bin/env bash
# ==============================================================================
# Wien2kGen Installer (v9.8.0) – Production-Grade HPC Setup Script
# Supports root/user installation, online/offline modes, safe cleanup,
# automatic verification, and seamless integration with pyproject.toml
# ==============================================================================
set -euo pipefail

# Configuration
APP_NAME="wien2k_gen"
APP_VERSION="9.8.0"
OFFLINE_DIR="offline_packages"
BINARIES=("wien2k_gen" "wien2k_sbatch" "wien2k_wizard")

# Colors & Logging
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()    { echo -e "${CYAN}[INFO]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# CLI Flags
UNINSTALL=false
DRY_RUN=false
FORCE=false
PREFIX_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --uninstall) UNINSTALL=true; shift ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --force)     FORCE=true; shift ;;
    --prefix=*)  PREFIX_OVERRIDE="${1#*=}"; shift ;;
    *)           shift ;;
  esac
done

# ==============================================================================
# 1. Determine Installation Prefix
# ==============================================================================
if [[ -n "$PREFIX_OVERRIDE" ]]; then
  INSTALL_PREFIX="$PREFIX_OVERRIDE"
  BIN_LINK_DIR="${INSTALL_PREFIX}/bin"
  PROFILE_FILE="${HOME}/.bashrc"
elif [[ "$(id -u)" -eq 0 ]]; then
  INSTALL_PREFIX="/opt/${APP_NAME}"
  BIN_LINK_DIR="/usr/local/bin"
  PROFILE_FILE="/etc/profile.d/${APP_NAME}.sh"
else
  INSTALL_PREFIX="${HOME}/.local/opt/${APP_NAME}"
  BIN_LINK_DIR="${HOME}/.local/bin"
  [[ -n "${ZSH_VERSION:-}" ]] && PROFILE_FILE="${HOME}/.zshrc" || PROFILE_FILE="${HOME}/.bashrc"
fi

LIB_DIR="${INSTALL_PREFIX}/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"

log "📦 Install Prefix: ${INSTALL_PREFIX}"
log "🔗 Bin Directory:  ${BIN_LINK_DIR}"

# ==============================================================================
# 2. Uninstall Mode
# ==============================================================================
if $UNINSTALL; then
  log "🧹 Uninstalling ${APP_NAME} from ${INSTALL_PREFIX}..."
  [[ -d "$INSTALL_PREFIX" ]] && rm -rf "$INSTALL_PREFIX"
  for bin in "${BINARIES[@]}"; do rm -f "${BIN_LINK_DIR}/${bin}" 2>/dev/null || true; done
  [[ -f "$PROFILE_FILE" ]] && sed -i "/${APP_NAME}/d" "$PROFILE_FILE" 2>/dev/null || true
  success "✅ ${APP_NAME} successfully uninstalled. Restart shell or run: source ${PROFILE_FILE}"
  exit 0
fi

# ==============================================================================
# 3. Existing Installation Check
# ==============================================================================
if [[ -d "${INSTALL_PREFIX}/bin" ]]; then
  if ! $FORCE; then
    warn "⚠️  Previous installation detected at ${INSTALL_PREFIX}."
    read -rp "❓ Overwrite and proceed? [y/N]: " confirm
    [[ "${confirm}" =~ ^[Yy]$ ]] || exit 0
  fi
  log "🧹 Cleaning previous installation..."
  rm -rf "${INSTALL_PREFIX}"
  for bin in "${BINARIES[@]}"; do rm -f "${BIN_LINK_DIR}/${bin}" 2>/dev/null || true; done
  [[ -f "$PROFILE_FILE" ]] && sed -i "/${APP_NAME}/d" "$PROFILE_FILE" 2>/dev/null || true
fi

# ==============================================================================
# 4. Internet Connectivity Check (HPC-safe)
# ==============================================================================
check_internet() {
  # Try ping, curl, then python urllib as fallback
  ping -c 1 -W 2 pypi.org >/dev/null 2>&1 && return 0
  curl -s --head -m 3 https://pypi.org >/dev/null 2>&1 && return 0
  python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=3)" 2>/dev/null && return 0
  return 1
}

USE_ONLINE=false
if check_internet; then
  log "🌐 Internet detected."
  if $DRY_RUN; then USE_ONLINE=true
  else
    read -rp "❓ Download packages matching this Python version? [Y/n]: " ans
    [[ "${ans}" =~ ^[Nn]$ ]] || USE_ONLINE=true
  fi
else
  log "📴 No internet connection."
  USE_ONLINE=false
fi

# ==============================================================================
# 5. Installation
# ==============================================================================
INSTALL_CMD="python3 -m pip install --prefix='${INSTALL_PREFIX}' --no-cache-dir --no-warn-script-location"

if $USE_ONLINE; then
  log "⬇️  Installing from PyPI..."
  if $DRY_RUN; then
    log "🔍 Dry-run: ${INSTALL_CMD} ."
  else
    ${INSTALL_CMD} .
  fi
else
  if [[ ! -d "${OFFLINE_DIR}" ]] || [[ -z "$(ls -A "${OFFLINE_DIR}" 2>/dev/null)" ]]; then
    error "❌ Offline directory '${OFFLINE_DIR}/' is missing or empty. Cannot proceed."
  fi
  log "📦 Installing from offline packages..."
  if $DRY_RUN; then
    log "🔍 Dry-run: ${INSTALL_CMD} --find-links='${OFFLINE_DIR}' --no-index ."
  else
    ${INSTALL_CMD} --find-links="${OFFLINE_DIR}" --no-index .
  fi
fi
success "✅ Package installed."

# ==============================================================================
# 6. Symlinks & Environment Setup
# ==============================================================================
log "🔗 Creating symlinks in ${BIN_LINK_DIR}..."
mkdir -p "${BIN_LINK_DIR}"
for bin in "${BINARIES[@]}"; do
  target="${INSTALL_PREFIX}/bin/${bin}"
  [[ -f "$target" ]] && ln -sf "$target" "${BIN_LINK_DIR}/${bin}" || warn "⚠️  Missing binary: ${bin}"
done

# PYTHONPATH setup
export_line="export PYTHONPATH=\"${LIB_DIR}:\${PYTHONPATH}\""
if ! grep -qF "${LIB_DIR}" "${PROFILE_FILE}" 2>/dev/null; then
  echo -e "\n# ${APP_NAME} v${APP_VERSION} Environment" >> "${PROFILE_FILE}"
  echo "${export_line}" >> "${PROFILE_FILE}"
  log "📝 Added PYTHONPATH to ${PROFILE_FILE}. Run: source ${PROFILE_FILE}"
else
  log "📝 PYTHONPATH already configured."
fi

# ==============================================================================
# 7. Post-Install Verification
# ==============================================================================
log "🧪 Running verification..."
export PYTHONPATH="${LIB_DIR}:${PYTHONPATH:-}"
BIN_PATH="${BIN_LINK_DIR}/wien2k_gen"

if [[ ! -x "$BIN_PATH" ]]; then
  error "❌ Binary not found at ${BIN_PATH}. Installation may have failed."
fi

VER_OUT=$("${BIN_PATH}" --version 2>/dev/null) || true
if [[ "${VER_OUT}" == *"${APP_VERSION}"* ]]; then
  success "✅ Version check passed: ${VER_OUT}"
else
  warn "⚠️  Version mismatch. Output: ${VER_OUT}"
fi

if python3 -c "import ${APP_NAME}; print('Module import: OK')" >/dev/null 2>&1; then
  success "✅ Python module import passed."
else
  error "❌ Module import failed. Check PYTHONPATH or dependencies."
fi

success "🎉 Installation completed successfully!"
log "📌 Next steps:"
log "   1. source ${PROFILE_FILE}"
log "   2. ${APP_NAME} --help"
log "   3. ${APP_NAME} tui"