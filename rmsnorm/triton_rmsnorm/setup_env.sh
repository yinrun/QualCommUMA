#!/bin/bash
# ===- setup_env.sh --------------------------------------------------------===
#
# Environment setup for Triton RMSNorm on Hexagon via hexagon-mlir.
#
# This script:
#   1. Sets required environment variables
#   2. Builds the hexagon-mlir toolchain (LLVM + Triton + hexagon backend)
#   3. Activates the Python virtual environment
#
# Prerequisites:
#   - Hexagon SDK 6.4+
#   - Android device connected via adb (for on-device execution)
#   - ~50 GB disk space for LLVM build
#
# Usage:
#   source setup_env.sh          # full build (first time)
#   source setup_env.sh --skip-build   # env only (after initial build)
#
# ===------------------------------------------------------------------------===

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/../hexagon-mlir"
BASE_DIR="$(cd "${REPO_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Hexagon architecture version
# SM8850 = Hexagon V81. hexagon-mlir FAQ lists v73/v75/v79 as supported.
# Try v81 first; fall back to v79 if compilation fails.
# ---------------------------------------------------------------------------
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH_VERSION:-81}"
echo "HEXAGON_ARCH_VERSION=${HEXAGON_ARCH_VERSION}"

# ---------------------------------------------------------------------------
# Android device for on-device test execution
# ---------------------------------------------------------------------------
export ANDROID_HOST="${ANDROID_HOST:-}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-}"
if [ -n "${ANDROID_SERIAL}" ]; then
    echo "ANDROID_SERIAL=${ANDROID_SERIAL}"
fi

# ---------------------------------------------------------------------------
# Build toolchain (calls hexagon-mlir's build script)
# ---------------------------------------------------------------------------
if [ "${1:-}" != "--skip-build" ]; then
    echo "Building hexagon-mlir toolchain..."
    echo "This will download LLVM, Hexagon SDK/Tools, and build everything."
    echo "Estimated time: 30-60 minutes on first run."
    echo ""
    cd "${REPO_DIR}"
    bash scripts/build_hexagon_mlir.sh
    cd "${SCRIPT_DIR}"
else
    echo "Skipping build (--skip-build)"
fi

# ---------------------------------------------------------------------------
# Activate virtual environment (created by build_hexagon_mlir.sh)
# ---------------------------------------------------------------------------
VENV_DIR="${BASE_DIR}/mlir-env"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: ${VENV_DIR}"
else
    echo "WARNING: Virtual environment not found at ${VENV_DIR}"
    echo "Run without --skip-build first to create it."
fi

# ---------------------------------------------------------------------------
# Set paths that the build script exports
# ---------------------------------------------------------------------------
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-${BASE_DIR}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/}"
export HEXAGON_TOOLS="${HEXAGON_TOOLS:-${BASE_DIR}/HEXAGON_TOOLS/Tools}"
export HEXKL_ROOT="${HEXKL_ROOT:-${BASE_DIR}/HEXKL_DIR/hexkl_addon}"
export LLVM_PROJECT_BUILD_DIR="${LLVM_PROJECT_BUILD_DIR:-${BASE_DIR}/LLVM_DIR/llvm-project/build}"

echo ""
echo "Environment ready. Run tests with:"
echo "  cd ${SCRIPT_DIR}"
echo "  pytest -sv rmsnorm_kernel.py"
echo ""
echo "Or run specific scenarios:"
echo "  pytest -sv rmsnorm_kernel.py -k 'decode and fp16 and 1T'"
