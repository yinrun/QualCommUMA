# QNN HTP 32MB Add Demo

这个 demo 在手机端通过 QNN HTP 后端执行两个 128MB 张量（int8）的逐元素相加，并验证输出是否为 3。
当前张量显式使用 NHWC 维度 `[N,H,W,C]`（`N=8, H=1, W=1, C=kTileElements/8`）。
为满足 HTP VTCM 限制，计算会按 1MB tile 分块执行，但总体处理的数据仍是 32MB。

## 目录结构

- `src/main.cpp`: 直接用 QNN C API 构建图并执行
- `build_android.sh`: NDK 交叉编译脚本
- `run_on_device.sh`: 推送依赖并在设备上运行

## 前置条件

- Android SDK 已安装（路径：`/Users/yinrun/Library/Android/sdk`，包含 NDK）
- QNN SDK 路径：`/Users/yinrun/Workspace/qairt/2.40.0.251030`
- 手机已连接，`adb devices` 可见

## 编译

```bash
cd /Users/yinrun/Workspace/project/QualCommGraph/demo_qnn_htp_add
export ANDROID_SDK_ROOT=/Users/yinrun/Library/Android/sdk
export QNN_SDK_ROOT=/Users/yinrun/Workspace/qairt/2.40.0.251030
./build_android.sh
```

## 运行（HTP v81，适用于 SM8850）

```bash
export QNN_SDK_ROOT=/Users/yinrun/Workspace/qairt/2.40.0.251030
./run_on_device.sh
```

输出示例：

```
[QNN] HTP Add demo starting...
[QNN] Done. max_error=0 sample_out=3
```

## 如果 HTP 版本不一致

SM8850 对应 HTP v81。若设备提示找不到 skel/so 或执行失败：

1. 先运行 `qnn-platform-validator` 获取目标 HTP 版本。
2. 修改 `run_on_device.sh` 中 `hexagon-v81` 和 `V81` 的库为对应版本（例如 v68/v69/v73/v75/v79）。

## 说明

- 每个输入张量大小：`8,388,608 * 4B = 32MB`
- 运算类型：`ElementWiseAdd`（`QNN_OP_ELEMENT_WISE_ADD`）

## user 版本设备提示

在 `user` 版本设备上，HTP 通常只允许加载 `/vendor/lib/rfsa/adsp` 下的签名 DSP 库，
同时 QNN 运行时建议优先用 `/vendor/lib64` 的 `libQnnHtp.so` 等库以避免版本不兼容。
脚本会优先使用这些路径；若没有找到再回退到 SDK 的 `unsigned` 库。
