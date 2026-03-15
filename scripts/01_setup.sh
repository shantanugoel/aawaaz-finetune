#!/usr/bin/env bash
# 01_setup.sh — Create venv with uv and install dependencies for the
# detected (or specified) platform.
#
# Usage:
#   ./scripts/01_setup.sh              # auto-detect platform
#   ./scripts/01_setup.sh --platform linux
#   ./scripts/01_setup.sh --platform mac
#
# This script is run manually — it is NOT called by the pipeline orchestrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parse args ──────────────────────────────────────────────────────────────

PLATFORM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --platform requires a value (linux or mac)."
                exit 1
            fi
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--platform linux|mac]"
            echo ""
            echo "Creates a Python virtual environment and installs dependencies."
            echo "If --platform is omitted, auto-detects from the OS."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── Detect platform ────────────────────────────────────────────────────────

if [[ -z "$PLATFORM" ]]; then
    case "$(uname -s)" in
        Darwin*)  PLATFORM="mac" ;;
        Linux*)   PLATFORM="linux" ;;
        *)
            echo "ERROR: Could not auto-detect platform from '$(uname -s)'."
            echo "       Please specify --platform linux or --platform mac."
            exit 1
            ;;
    esac
    echo "Auto-detected platform: ${PLATFORM}"
fi

if [[ "$PLATFORM" != "linux" && "$PLATFORM" != "mac" ]]; then
    echo "ERROR: Invalid platform '${PLATFORM}'. Must be 'linux' or 'mac'."
    exit 1
fi

REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements-${PLATFORM}.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "ERROR: Requirements file not found: ${REQUIREMENTS_FILE}"
    exit 1
fi

# ── Check for uv ────────────────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── Create venv ─────────────────────────────────────────────────────────────

VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
    echo "To recreate, delete it first: rm -rf ${VENV_DIR}"
else
    echo "Creating virtual environment at ${VENV_DIR} ..."
    uv venv "${VENV_DIR}"
fi

# ── Install dependencies ───────────────────────────────────────────────────

echo ""
echo "Installing dependencies from ${REQUIREMENTS_FILE} ..."
uv pip install --python "${VENV_DIR}/bin/python" -r "${REQUIREMENTS_FILE}"

# ── Create directory structure ──────────────────────────────────────────────

echo ""
echo "Creating project directories ..."
mkdir -p \
    "${PROJECT_ROOT}/data/raw" \
    "${PROJECT_ROOT}/data/synthetic/rejected" \
    "${PROJECT_ROOT}/data/combined" \
    "${PROJECT_ROOT}/data/eval" \
    "${PROJECT_ROOT}/models/base" \
    "${PROJECT_ROOT}/models/adapters" \
    "${PROJECT_ROOT}/models/fused" \
    "${PROJECT_ROOT}/models/mlx" \
    "${PROJECT_ROOT}/models/quantized" \
    "${PROJECT_ROOT}/eval_results"

# ── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "✅  Setup complete (platform: ${PLATFORM})."
echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Then run the pipeline:"
echo "  python scripts/run_pipeline.py --all"
