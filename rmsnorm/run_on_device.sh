#!/usr/bin/env bash
set -euo pipefail

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"

DIR="/data/local/tmp/rmsnorm_test"
LIB="${DIR}/lib"
HTP="${DIR}/htp"

adb shell "mkdir -p ${DIR}/kernels ${LIB} ${HTP}"

adb push build/android/rmsnorm_test "${DIR}/"
adb push kernels/rmsnorm.cl "${DIR}/kernels/"

# ARM64 QNN libs
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so" "${LIB}/"
adb push "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81CalculatorStub.so" "${LIB}/"

# Hexagon V81 DSP libs
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libCalculator_skel.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSystem.so" "${HTP}/"
adb push "${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnSaver.so" "${HTP}/"

adb shell "chmod 755 ${DIR}/rmsnorm_test"

adb shell "cd ${DIR} && \
  export LD_LIBRARY_PATH=${LIB}:/vendor/lib64:\$LD_LIBRARY_PATH && \
  export ADSP_LIBRARY_PATH=${HTP} && \
  ./rmsnorm_test $*"
