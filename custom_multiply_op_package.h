#ifndef CUSTOM_MULTIPLY_OP_PACKAGE_H
#define CUSTOM_MULTIPLY_OP_PACKAGE_H

#include "QnnOpPackage.h"
#include "CPU/QnnCpuOpPackage.h"

#ifdef __cplusplus
extern "C" {
#endif

// OpPackage 名称常量
#define CUSTOM_MULTIPLY_OP_PACKAGE_NAME "CustomMultiplyOpPackage"
#define CUSTOM_MULTIPLY_OP_NAME "CustomMultiply"

// OpPackage 接口提供者函数
Qnn_ErrorHandle_t CustomMultiplyOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface);

// 直接调用自定义算子的包装函数（用于简化调用）
Qnn_ErrorHandle_t CustomMultiplyOp_execute(QnnCpuOpPackage_Node_t* node);

#ifdef __cplusplus
}
#endif

#endif // CUSTOM_MULTIPLY_OP_PACKAGE_H

