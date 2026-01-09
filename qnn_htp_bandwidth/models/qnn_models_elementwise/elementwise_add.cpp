/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-pytorch-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=False; apply_masked_softmax=uncompressed; arch_checker=False; backend=None; bias_bitwidth=8; calc_static_encodings=False; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=None; disable_batchnorm_folding=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=False; dumpIR=False; dump_encoding_json=False; dump_ir=; dump_ir_optimizer_config_template=False; dump_optimization_pass_mode_config=False; dump_out_names=False; dump_pass_trace_info=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_relay=None; enable_framework_trace=False; enable_match_gathernd=False; enable_match_topk=False; enable_per_row_quantized_bias=False; exclude_named_tensors=False; expand_gru_op_structure=False; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=8; float_bias_bw=0; float_bitwidth=8; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=False; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=[['input', '1,67108864']]; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=input_list_elementwise.txt; input_type=[]; ir_optimizer_config=; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; op_package_lib=; optimization_pass_mode=ir_optimizer_mainline; optimization_pass_mode_config=; out_names=['aten__add_0.add_0_0']; overwrite_model_prefix=True; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[]; preserve_onnx_output_order=False; pytorch_custom_op_lib=; quantization_overrides=; quantizer_log=None; quantizer_log_level=LogLevel.NONE; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=False; unroll_lstm_time_steps=False; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; use_quantize_v2=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.40.0.251030114326_189385";
extern "C" {
static ModelError_t addTensor_input(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input[] = {1, 67108864};
  VALIDATE(model.addTensor("input", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                 .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                    {.scaleOffsetEncoding= {.scale= 0.0429119765758514404296875000000000000000f, .offset= -129}}},
                                 .rank= 2,
                                 .dimensions=dimensions_input,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_add_value(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_add_value[] = {1};
  VALIDATE(model.addTensor("add_value", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "add_value",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                 .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                    {.scaleOffsetEncoding= {.scale= 0.0039215688593685626983642578125000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_add_value,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(add_value),
                                                .dataSize=BINLEN(add_value)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_aten__add_0_add_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR aten__add_0_add_0 */
  Qnn_Param_t params_aten__add_0_add_0[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_aten__add_0_add_0[] = {
    "input",
    "add_value"
  };
  uint32_t dimensions_aten__add_0_add_0_0[] = {1, 67108864};
  Qnn_Tensor_t outputs_aten__add_0_add_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "aten__add_0_add_0_0",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0429119765758514404296875000000000000000f, .offset= -105}}},
            .rank= 2,
            .dimensions=dimensions_aten__add_0_add_0_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "aten__add_0_add_0", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_aten__add_0_add_0, // Node Params
                         1, // Num Node Params
                         inputs_aten__add_0_add_0, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_aten__add_0_add_0, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

QNN_API
ModelError_t ElementwiseAdd_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                          QNN_INTERFACE_VER_TYPE interface,
                                          Qnn_ContextHandle_t contextHandle,
                                          const GraphConfigInfo_t** graphsConfigInfo,
                                          const uint32_t numGraphsConfigInfo,
                                          GraphInfoPtr_t** graphsInfo,
                                          uint32_t* numGraphsInfo,
                                          bool debug,
                                          QnnLog_Callback_t logCallback,
                                          QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for elementwise_add*/
  QnnModel elementwise_add;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("elementwise_add", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(elementwise_add.initialize(backendHandle, interface, contextHandle, "elementwise_add", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_input(elementwise_add), err);
  VALIDATE(addTensor_add_value(elementwise_add), err);
  VALIDATE(addNode_aten__add_0_add_0(elementwise_add), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&elementwise_add};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t ElementwiseAdd_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}