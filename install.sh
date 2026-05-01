#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install.sh — Install rwkz from GitHub Releases
# ---------------------------------------------------------------------------
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/LuRenJiasWorld/RWKZ/master/install.sh | bash
#
#   INSTALL_DIR=~/bin ./install.sh   # custom install location
#   VERSION=0.2.0 ./install.sh       # pin a specific version
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="LuRenJiasWorld/RWKZ"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

# ─── API helpers ───────────────────────────────────────────────────────────

get_latest_version() {
    local url="https://api.github.com/repos/${REPO}/releases/latest"
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/'
    elif command -v wget &>/dev/null; then
        wget -qO- "$url" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/'
    fi
}

# ─── Detect platform ──────────────────────────────────────────────────────

detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux)
            os="unknown-linux-gnu"
            case "$(uname -m)" in
                x86_64|amd64)  arch="x86_64" ;;
                aarch64|arm64) arch="aarch64" ;;
                *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
            esac
            ;;
        Darwin)
            os="apple-darwin"
            case "$(uname -m)" in
                x86_64|amd64)  arch="x86_64" ;;
                aarch64|arm64) arch="aarch64" ;;
                *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
            esac
            ;;
        *)
            echo "Unsupported OS: $(uname -s)" >&2
            echo "Build from source: https://github.com/${REPO}" >&2
            exit 1
            ;;
    esac

    echo "${arch}-${os}"
}

# ─── Main ─────────────────────────────────────────────────────────────────

VERSION="${VERSION:-$(get_latest_version)}"
if [ -z "$VERSION" ]; then
    echo "Error: could not determine latest version. Set VERSION= manually." >&2
    exit 1
fi

PLATFORM=$(detect_platform)
ARCHIVE="rwkz-v${VERSION}-${PLATFORM}"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ARCHIVE}"

echo "==> Installing rwkz v${VERSION} for ${PLATFORM}"
echo "    ${URL}"

# Download
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if command -v curl &>/dev/null; then
    curl -fsSL "$URL" -o "${TMP_DIR}/rwkz"
elif command -v wget &>/dev/null; then
    wget -q "$URL" -O "${TMP_DIR}/rwkz"
else
    echo "Error: curl or wget required" >&2
    exit 1
fi

# Install
mkdir -p "$INSTALL_DIR"
chmod +x "${TMP_DIR}/rwkz"
mv "${TMP_DIR}/rwkz" "${INSTALL_DIR}/rwkz"

echo "==> Installed to ${INSTALL_DIR}/rwkz"

# Verify
if "${INSTALL_DIR}/rwkz" --version &>/dev/null 2>&1; then
    echo "    $("${INSTALL_DIR}/rwkz" --version)"
else
    echo "    Warning: binary may have runtime issues (libc mismatch?)."
    echo "    Try building from source: https://github.com/${REPO}"
fi

# PATH hint
if ! echo "$PATH" | tr ':' '\n' | grep -qxF "$INSTALL_DIR"; then
    echo ""
    echo "Add ${INSTALL_DIR} to your PATH:"
    echo "    export PATH=\"${INSTALL_DIR}:\$PATH\"    # add to ~/.bashrc or ~/.zshrc"
fi

echo ""
echo "Try it: rwkz compress hello.txt hello.rkz --q Q4_K_M"
