#!/usr/bin/env bash
set -euo pipefail
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"

DEVICE_DIR="/data/local/tmp/sync_op_test"
LIB_DIR="${DEVICE_DIR}/lib"
HTP_DIR="${DEVICE_DIR}/htp"

adb shell "mkdir -p ${LIB_DIR} ${HTP_DIR}"

# Push test binary
adb push build/android/sync_op_test "${DEVICE_DIR}/"

# QNN runtime libraries
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so" "${LIB_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81CalculatorStub.so" "${LIB_DIR}/"

# Hexagon V81 skel libraries
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libCalculator_skel.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSystem.so" "${HTP_DIR}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSaver.so" "${HTP_DIR}/"

# HeteroEdge op package (SyncWait + RmsNorm)
HETEROEDGE_DIR="../fast_sync_test/heteroedge_op/build"
if [ -d "${HETEROEDGE_DIR}" ]; then
  adb push "${HETEROEDGE_DIR}/aarch64-android/libQnnHtpHeteroEdgeOpPackage.so" "${DEVICE_DIR}/"
  adb push "${HETEROEDGE_DIR}/hexagon-v81/libQnnHtpHeteroEdgeOpPackage.so" "${HTP_DIR}/"
else
  echo "WARNING: heteroedge_op not found at ${HETEROEDGE_DIR}"
  echo "  Build it first: cd ../fast_sync_test/heteroedge_op && bash build.sh"
fi

adb shell "chmod 755 ${DEVICE_DIR}/sync_op_test"

adb shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${LIB_DIR}:/vendor/lib64:\$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=${HTP_DIR} && \
  ./sync_op_test $*"
