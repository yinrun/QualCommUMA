#include "custom_multiply_op_package.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 自定义乘法算子的实现
// 乘数硬编码在 kernel 内（3.0）
static Qnn_ErrorHandle_t custom_multiply_op_impl_internal(QnnCpuOpPackage_Node_t* node) {
    if (node->numOfInputs != 1 || node->numOfOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    
    QnnCpuOpPackage_Tensor_t* input = node->inputs[0];
    QnnCpuOpPackage_Tensor_t* output = node->outputs[0];
    
    if (input->dataType != QNN_CPU_DATATYPE_FLOAT_32 || 
        output->dataType != QNN_CPU_DATATYPE_FLOAT_32) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    
    const float* in_data = (const float*)input->data;
    float* out_data = (float*)output->data;
    
    // 计算元素数量
    size_t num_elements = 1;
    for (uint32_t i = 0; i < input->rank; i++) {
        num_elements *= input->currentDimensions[i];
    }
    
    // 执行乘法：每个值乘以 3.0（乘数硬编码在 kernel 内）
    const float MULTIPLIER = 4.0f;
    for (size_t i = 0; i < num_elements; i++) {
        out_data[i] = in_data[i] * MULTIPLIER;
    }
    
    return QNN_SUCCESS;
}

// 直接调用自定义算子的包装函数（用于简化调用）
Qnn_ErrorHandle_t CustomMultiplyOp_execute(QnnCpuOpPackage_Node_t* node) {
    return custom_multiply_op_impl_internal(node);
}

// OpPackage 接口实现
static bool sg_packageInitialized = false;
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = NULL;

static const char* sg_packageName = "CustomMultiplyOpPackage";
static const char* sg_opName = "CustomMultiply";
static const char* sg_opNames[] = {sg_opName};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_CPU_API_VERSION_INIT;
static Qnn_Version_t sg_opsetVersion = {1, 0, 0};

static QnnOpPackage_Info_t sg_packageInfo = {
    sg_packageName,
    sg_opNames,
    NULL,
    1,  // numOperations
    NULL,  // optimizations
    0,  // numOptimizations
    NULL,  // sdkBuildId
    &sg_sdkApiVersion,
    NULL,  // packageInfo
    &sg_opsetVersion,
    {0}  // reserved
};

static Qnn_ErrorHandle_t customOpPackage_init(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    }
    sg_globalInfra = infrastructure;
    sg_packageInitialized = true;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t customOpPackage_getInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }
    if (!info) {
        return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    }
    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t customOpPackage_validateOpConfig(Qnn_OpConfig_t opConfig) {
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }
    if (strcmp(opConfig.v1.packageName, sg_packageName) != 0) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    if (strcmp(opConfig.v1.typeName, sg_opName) != 0) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    if (opConfig.v1.numOfInputs != 1 || opConfig.v1.numOfOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t customOpPackage_createOpImpl(
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* opImplPtr) {
    (void)graphInfrastructure;
    
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }
    
    // 创建 OpImpl
    QnnCpuOpPackage_OpImpl_t* opImpl = (QnnCpuOpPackage_OpImpl_t*)malloc(sizeof(QnnCpuOpPackage_OpImpl_t));
    if (!opImpl) {
        return QNN_OP_PACKAGE_ERROR_GENERAL;
    }
    
    // 创建包装函数
    static auto wrapper = [](void* opPkgNodeData) -> Qnn_ErrorHandle_t {
        return custom_multiply_op_impl_internal((QnnCpuOpPackage_Node_t*)opPkgNodeData);
    };
    opImpl->opImplFn = wrapper;
    opImpl->userData = node;
    
    *opImplPtr = (QnnOpPackage_OpImpl_t)opImpl;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t customOpPackage_freeOpImpl(QnnOpPackage_OpImpl_t opImpl) {
    if (opImpl) {
        free(opImpl);
    }
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t customOpPackage_terminate() {
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }
    sg_globalInfra = NULL;
    sg_packageInitialized = false;
    return QNN_SUCCESS;
}

// OpPackage 接口提供者
extern "C" Qnn_ErrorHandle_t CustomMultiplyOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface) {
    if (!interface) {
        return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    }
    
    interface->interfaceVersion = {1, 4, 0};
    interface->v1_4.init = customOpPackage_init;
    interface->v1_4.terminate = customOpPackage_terminate;
    interface->v1_4.getInfo = customOpPackage_getInfo;
    interface->v1_4.validateOpConfig = customOpPackage_validateOpConfig;
    interface->v1_4.createOpImpl = customOpPackage_createOpImpl;
    interface->v1_4.freeOpImpl = customOpPackage_freeOpImpl;
    interface->v1_4.logInitialize = NULL;
    interface->v1_4.logSetLevel = NULL;
    interface->v1_4.logTerminate = NULL;
    
    return QNN_SUCCESS;
}

