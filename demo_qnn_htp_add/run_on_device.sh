#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  echo "QNN_SDK_ROOT is not set"
  exit 1
fi

DEVICE_DIR="/data/local/tmp/qnn_htp_add"
LIB_DIR="${DEVICE_DIR}/lib"
HTP_DIR="${DEVICE_DIR}/htp"
VENDOR_HTP_DIR="/vendor/lib/rfsa/adsp"
VENDOR_LIB64_DIR="/vendor/lib64"

adb shell "mkdir -p ${LIB_DIR} ${HTP_DIR}"

adb push build/android/qnn_htp_add_demo "${DEVICE_DIR}/"

if adb shell "test -f ${VENDOR_LIB64_DIR}/libQnnHtp.so"; then
  QNN_LIB_PATH="${VENDOR_LIB64_DIR}"
else
  adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${LIB_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${LIB_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${LIB_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so" "${LIB_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81CalculatorStub.so" "${LIB_DIR}/"
  QNN_LIB_PATH="${LIB_DIR}"
fi

# Prefer vendor-signed DSP libs on user builds.
if adb shell "test -f ${VENDOR_HTP_DIR}/libQnnHtpV81Skel.so"; then
  HTP_LIB_PATH="${VENDOR_HTP_DIR}"
else
  adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "${HTP_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so" "${HTP_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libCalculator_skel.so" "${HTP_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSystem.so" "${HTP_DIR}/"
  adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSaver.so" "${HTP_DIR}/"
  HTP_LIB_PATH="${HTP_DIR}"
fi

adb shell "chmod 755 ${DEVICE_DIR}/qnn_htp_add_demo"

adb shell "export LD_LIBRARY_PATH=${QNN_LIB_PATH}:\$LD_LIBRARY_PATH; \
  export ADSP_LIBRARY_PATH=${HTP_LIB_PATH}; \
  ${DEVICE_DIR}/qnn_htp_add_demo"
