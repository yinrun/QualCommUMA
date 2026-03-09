#!/usr/bin/env bash
set -euo pipefail

# Use same SDK paths as rmsnorm/custom_op
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/home/yinrun/workspace/HeteroEdge/rmsnorm/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}"
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/home/yinrun/Android/Sdk/android-ndk-r25c}"

echo "=== Building SyncWait HTP Op Package ==="
echo "QNN_SDK_ROOT:     ${QNN_SDK_ROOT}"
echo "HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}"
echo "ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT}"
echo ""

cd "$(dirname "$0")"

make clean
make htp_v81 htp_aarch64 -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Hexagon V81 : build/hexagon-v81/libQnnHtpSyncWaitOpPackage.so"
echo "AArch64 ARM : build/aarch64-android/libQnnHtpSyncWaitOpPackage.so"
