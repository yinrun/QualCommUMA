/*******************************************************************************
 * Copyright (c) 2008-2023 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef OPENCL_CL_EXT_H_
#define OPENCL_CL_EXT_H_

/*
** This header is generated from the Khronos OpenCL XML API Registry.
*/

#include <CL/cl.h>

/* CL_NO_PROTOTYPES implies CL_NO_EXTENSION_PROTOTYPES: */
#if defined(CL_NO_PROTOTYPES) && !defined(CL_NO_EXTENSION_PROTOTYPES)
#define CL_NO_EXTENSION_PROTOTYPES
#endif

/* CL_NO_EXTENSION_PROTOTYPES implies
   CL_NO_ICD_DISPATCH_EXTENSION_PROTOTYPES and
   CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES: */
#if defined(CL_NO_EXTENSION_PROTOTYPES) && \
    !defined(CL_NO_ICD_DISPATCH_EXTENSION_PROTOTYPES)
#define CL_NO_ICD_DISPATCH_EXTENSION_PROTOTYPES
#endif
#if defined(CL_NO_EXTENSION_PROTOTYPES) && \
    !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)
#define CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES
#endif

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************
* cl_khr_command_buffer (beta)
***************************************************************/
#if defined(CL_ENABLE_BETA_EXTENSIONS)

#define cl_khr_command_buffer 1
#define CL_KHR_COMMAND_BUFFER_EXTENSION_NAME \
    "cl_khr_command_buffer"


#define CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION CL_MAKE_VERSION(0, 9, 8)

typedef cl_bitfield         cl_device_command_buffer_capabilities_khr;
typedef struct _cl_command_buffer_khr* cl_command_buffer_khr;
typedef cl_uint             cl_sync_point_khr;
typedef cl_uint             cl_command_buffer_info_khr;
typedef cl_uint             cl_command_buffer_state_khr;
typedef cl_properties       cl_command_buffer_properties_khr;
typedef cl_bitfield         cl_command_buffer_flags_khr;
typedef cl_properties       cl_command_properties_khr;
typedef struct _cl_mutable_command_khr* cl_mutable_command_khr;

/* cl_device_info */
#define CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR           0x12A9
#define CL_DEVICE_COMMAND_BUFFER_SUPPORTED_QUEUE_PROPERTIES_KHR 0x129A
#define CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR 0x12AA

/* cl_device_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR      (1 << 0)
#define CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR (1 << 1)

/* cl_command_buffer_properties_khr */
#define CL_COMMAND_BUFFER_FLAGS_KHR                         0x1293

/* Error codes */
#define CL_INVALID_COMMAND_BUFFER_KHR                       -1138
#define CL_INVALID_SYNC_POINT_WAIT_LIST_KHR                 -1139
#define CL_INCOMPATIBLE_COMMAND_QUEUE_KHR                   -1140

/* cl_command_buffer_info_khr */
#define CL_COMMAND_BUFFER_QUEUES_KHR                        0x1294
#define CL_COMMAND_BUFFER_NUM_QUEUES_KHR                    0x1295
#define CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR               0x1296
#define CL_COMMAND_BUFFER_STATE_KHR                         0x1297
#define CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR              0x1298
#define CL_COMMAND_BUFFER_CONTEXT_KHR                       0x1299

/* cl_command_buffer_state_khr */
#define CL_COMMAND_BUFFER_STATE_RECORDING_KHR               0
#define CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR              1

/* cl_command_type */
#define CL_COMMAND_COMMAND_BUFFER_KHR                       0x12A8


typedef cl_command_buffer_khr CL_API_CALL
clCreateCommandBufferKHR_t(
    cl_uint num_queues,
    const cl_command_queue* queues,
    const cl_command_buffer_properties_khr* properties,
    cl_int* errcode_ret);

typedef clCreateCommandBufferKHR_t *
clCreateCommandBufferKHR_fn ;

typedef cl_int CL_API_CALL
clFinalizeCommandBufferKHR_t(
    cl_command_buffer_khr command_buffer);

typedef clFinalizeCommandBufferKHR_t *
clFinalizeCommandBufferKHR_fn ;

typedef cl_int CL_API_CALL
clRetainCommandBufferKHR_t(
    cl_command_buffer_khr command_buffer);

typedef clRetainCommandBufferKHR_t *
clRetainCommandBufferKHR_fn ;

typedef cl_int CL_API_CALL
clReleaseCommandBufferKHR_t(
    cl_command_buffer_khr command_buffer);

typedef clReleaseCommandBufferKHR_t *
clReleaseCommandBufferKHR_fn ;

typedef cl_int CL_API_CALL
clEnqueueCommandBufferKHR_t(
    cl_uint num_queues,
    cl_command_queue* queues,
    cl_command_buffer_khr command_buffer,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

typedef clEnqueueCommandBufferKHR_t *
clEnqueueCommandBufferKHR_fn ;

typedef cl_int CL_API_CALL
clCommandBarrierWithWaitListKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandBarrierWithWaitListKHR_t *
clCommandBarrierWithWaitListKHR_fn ;

typedef cl_int CL_API_CALL
clCommandCopyBufferKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandCopyBufferKHR_t *
clCommandCopyBufferKHR_fn ;

typedef cl_int CL_API_CALL
clCommandCopyBufferRectKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandCopyBufferRectKHR_t *
clCommandCopyBufferRectKHR_fn ;

typedef cl_int CL_API_CALL
clCommandCopyBufferToImageKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandCopyBufferToImageKHR_t *
clCommandCopyBufferToImageKHR_fn ;

typedef cl_int CL_API_CALL
clCommandCopyImageKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandCopyImageKHR_t *
clCommandCopyImageKHR_fn ;

typedef cl_int CL_API_CALL
clCommandCopyImageToBufferKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandCopyImageToBufferKHR_t *
clCommandCopyImageToBufferKHR_fn ;

typedef cl_int CL_API_CALL
clCommandFillBufferKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandFillBufferKHR_t *
clCommandFillBufferKHR_fn ;

typedef cl_int CL_API_CALL
clCommandFillImageKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandFillImageKHR_t *
clCommandFillImageKHR_fn ;

typedef cl_int CL_API_CALL
clCommandNDRangeKernelKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandNDRangeKernelKHR_t *
clCommandNDRangeKernelKHR_fn ;

typedef cl_int CL_API_CALL
clGetCommandBufferInfoKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_buffer_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef clGetCommandBufferInfoKHR_t *
clGetCommandBufferInfoKHR_fn ;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
clCreateCommandBufferKHR(
    cl_uint num_queues,
    const cl_command_queue* queues,
    const cl_command_buffer_properties_khr* properties,
    cl_int* errcode_ret) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clFinalizeCommandBufferKHR(
    cl_command_buffer_khr command_buffer) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandBufferKHR(
    cl_command_buffer_khr command_buffer) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandBufferKHR(
    cl_command_buffer_khr command_buffer) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCommandBufferKHR(
    cl_uint num_queues,
    cl_command_queue* queues,
    cl_command_buffer_khr command_buffer,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandBarrierWithWaitListKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandCopyBufferKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandCopyBufferRectKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandCopyBufferToImageKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandCopyImageKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandCopyImageToBufferKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandFillBufferKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandFillImageKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandNDRangeKernelKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetCommandBufferInfoKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_buffer_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) ;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/* From version 0.9.4 of the extension */

typedef cl_int CL_API_CALL
clCommandSVMMemcpyKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandSVMMemcpyKHR_t *
clCommandSVMMemcpyKHR_fn CL_API_SUFFIX__VERSION_2_0;

typedef cl_int CL_API_CALL
clCommandSVMMemFillKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef clCommandSVMMemFillKHR_t *
clCommandSVMMemFillKHR_fn CL_API_SUFFIX__VERSION_2_0;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) CL_API_SUFFIX__VERSION_2_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemFillKHR(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle) CL_API_SUFFIX__VERSION_2_0;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

#endif /* defined(CL_ENABLE_BETA_EXTENSIONS) */

/***************************************************************
* cl_khr_command_buffer_multi_device (beta)
***************************************************************/
#if defined(CL_ENABLE_BETA_EXTENSIONS)

#define cl_khr_command_buffer_multi_device 1
#define CL_KHR_COMMAND_BUFFER_MULTI_DEVICE_EXTENSION_NAME \
    "cl_khr_command_buffer_multi_device"


#define CL_KHR_COMMAND_BUFFER_MULTI_DEVICE_EXTENSION_VERSION CL_MAKE_VERSION(0, 9, 2)

typedef cl_bitfield         cl_platform_command_buffer_capabilities_khr;

/* cl_platform_info */
#define CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR         0x0908

/* cl_platform_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_PLATFORM_UNIVERSAL_SYNC_KHR       (1 << 0)
#define CL_COMMAND_BUFFER_PLATFORM_REMAP_QUEUES_KHR         (1 << 1)
#define CL_COMMAND_BUFFER_PLATFORM_AUTOMATIC_REMAP_KHR      (1 << 2)

/* cl_device_info */
#define CL_DEVICE_COMMAND_BUFFER_NUM_SYNC_DEVICES_KHR       0x12AB
#define CL_DEVICE_COMMAND_BUFFER_SYNC_DEVICES_KHR           0x12AC

/* cl_device_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_CAPABILITY_MULTIPLE_QUEUE_KHR     (1 << 4)

/* cl_command_buffer_flags_khr - bitfield */
#define CL_COMMAND_BUFFER_DEVICE_SIDE_SYNC_KHR              (1 << 2)


typedef cl_command_buffer_khr CL_API_CALL
clRemapCommandBufferKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_bool automatic,
    cl_uint num_queues,
    const cl_command_queue* queues,
    cl_uint num_handles,
    const cl_mutable_command_khr* handles,
    cl_mutable_command_khr* handles_ret,
    cl_int* errcode_ret);

typedef clRemapCommandBufferKHR_t *
clRemapCommandBufferKHR_fn ;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
clRemapCommandBufferKHR(
    cl_command_buffer_khr command_buffer,
    cl_bool automatic,
    cl_uint num_queues,
    const cl_command_queue* queues,
    cl_uint num_handles,
    const cl_mutable_command_khr* handles,
    cl_mutable_command_khr* handles_ret,
    cl_int* errcode_ret) ;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

#endif /* defined(CL_ENABLE_BETA_EXTENSIONS) */

/***************************************************************
* cl_khr_command_buffer_mutable_dispatch (beta)
***************************************************************/
#if defined(CL_ENABLE_BETA_EXTENSIONS)

#define cl_khr_command_buffer_mutable_dispatch 1
#define CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME \
    "cl_khr_command_buffer_mutable_dispatch"


#define CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_VERSION CL_MAKE_VERSION(0, 9, 5)

typedef cl_uint             cl_command_buffer_update_type_khr;
typedef cl_bitfield         cl_mutable_dispatch_fields_khr;
typedef cl_uint             cl_mutable_command_info_khr;
typedef struct _cl_mutable_dispatch_arg_khr {
    cl_uint arg_index;
    size_t arg_size;
    const void* arg_value;
} cl_mutable_dispatch_arg_khr;
typedef struct _cl_mutable_dispatch_exec_info_khr {
    cl_uint param_name;
    size_t param_value_size;
    const void* param_value;
} cl_mutable_dispatch_exec_info_khr;
typedef struct _cl_mutable_dispatch_config_khr {
    cl_mutable_command_khr command;
    cl_uint num_args;
    cl_uint num_svm_args;
    cl_uint num_exec_infos;
    cl_uint work_dim;
    const cl_mutable_dispatch_arg_khr* arg_list;
    const cl_mutable_dispatch_arg_khr* arg_svm_list;
    const cl_mutable_dispatch_exec_info_khr* exec_info_list;
    const size_t* global_work_offset;
    const size_t* global_work_size;
    const size_t* local_work_size;
} cl_mutable_dispatch_config_khr;
typedef cl_bitfield         cl_mutable_dispatch_asserts_khr;

/* cl_command_buffer_flags_khr - bitfield */
#define CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR              (1 << 0)
#define CL_COMMAND_BUFFER_MUTABLE_KHR                       (1 << 1)

/* cl_device_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR   (1 << 2)

/* Error codes */
#define CL_INVALID_MUTABLE_COMMAND_KHR                      -1141

/* cl_device_info */
#define CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR         0x12B0

/* cl_command_properties_khr */
#define CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR            0x12B1

/* cl_mutable_dispatch_fields_khr - bitfield */
#define CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR               (1 << 0)
#define CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR                 (1 << 1)
#define CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR                  (1 << 2)
#define CL_MUTABLE_DISPATCH_ARGUMENTS_KHR                   (1 << 3)
#define CL_MUTABLE_DISPATCH_EXEC_INFO_KHR                   (1 << 4)

/* cl_mutable_command_info_khr */
#define CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR                0x12A0
#define CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR               0x12A1
#define CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR                 0x12AD
#define CL_MUTABLE_COMMAND_PROPERTIES_ARRAY_KHR             0x12A2
#define CL_MUTABLE_DISPATCH_KERNEL_KHR                      0x12A3
#define CL_MUTABLE_DISPATCH_DIMENSIONS_KHR                  0x12A4
#define CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR          0x12A5
#define CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR            0x12A6
#define CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR             0x12A7

/* cl_command_buffer_update_type_khr */
#define CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR       0

/* cl_command_buffer_properties_khr */
#define CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR      0x12B7

/* cl_command_properties_khr */
#define CL_MUTABLE_DISPATCH_ASSERTS_KHR                     0x12B8

/* cl_mutable_dispatch_asserts_khr - bitfield */
#define CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR (1 << 0)


typedef cl_int CL_API_CALL
clUpdateMutableCommandsKHR_t(
    cl_command_buffer_khr command_buffer,
    cl_uint num_configs,
    const cl_command_buffer_update_type_khr* config_types,
    const void** configs);

typedef clUpdateMutableCommandsKHR_t *
clUpdateMutableCommandsKHR_fn ;

typedef cl_int CL_API_CALL
clGetMutableCommandInfoKHR_t(
    cl_mutable_command_khr command,
    cl_mutable_command_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef clGetMutableCommandInfoKHR_t *
clGetMutableCommandInfoKHR_fn ;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clUpdateMutableCommandsKHR(
    cl_command_buffer_khr command_buffer,
    cl_uint num_configs,
    const cl_command_buffer_update_type_khr* config_types,
    const void** configs) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetMutableCommandInfoKHR(
    cl_mutable_command_khr command,
    cl_mutable_command_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) ;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/* cl_command_buffer_state_khr */
#define CL_COMMAND_BUFFER_STATE_FINALIZED_KHR               2

#endif /* defined(CL_ENABLE_BETA_EXTENSIONS) */

/***************************************************************
* cl_khr_fp64
***************************************************************/
#define cl_khr_fp64 1
#define CL_KHR_FP64_EXTENSION_NAME \
    "cl_khr_fp64"


#define CL_KHR_FP64_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

#if !defined(CL_VERSION_1_2)
/* cl_device_info - defined in CL.h for OpenCL 1.2 and newer */
#define CL_DEVICE_DOUBLE_FP_CONFIG                          0x1032

#endif /* !defined(CL_VERSION_1_2) */

/***************************************************************
* cl_khr_fp16
***************************************************************/
#define cl_khr_fp16 1
#define CL_KHR_FP16_EXTENSION_NAME \
    "cl_khr_fp16"


#define CL_KHR_FP16_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

/* cl_device_info */
#define CL_DEVICE_HALF_FP_CONFIG                            0x1033

/***************************************************************
* cl_APPLE_SetMemObjectDestructor
***************************************************************/
#define cl_APPLE_SetMemObjectDestructor 1
#define CL_APPLE_SETMEMOBJECTDESTRUCTOR_EXTENSION_NAME \
    "cl_APPLE_SetMemObjectDestructor"


#define CL_APPLE_SETMEMOBJECTDESTRUCTOR_EXTENSION_VERSION CL_MAKE_VERSION(0, 0, 0)


typedef cl_int CL_API_CALL
clSetMemObjectDestructorAPPLE_t(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data);

typedef clSetMemObjectDestructorAPPLE_t *
clSetMemObjectDestructorAPPLE_fn CL_API_SUFFIX__VERSION_1_0;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clSetMemObjectDestructorAPPLE(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data) CL_API_SUFFIX__VERSION_1_0;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/***************************************************************
* cl_APPLE_ContextLoggingFunctions
***************************************************************/
#define cl_APPLE_ContextLoggingFunctions 1
#define CL_APPLE_CONTEXTLOGGINGFUNCTIONS_EXTENSION_NAME \
    "cl_APPLE_ContextLoggingFunctions"


#define CL_APPLE_CONTEXTLOGGINGFUNCTIONS_EXTENSION_VERSION CL_MAKE_VERSION(0, 0, 0)


typedef void CL_API_CALL
clLogMessagesToSystemLogAPPLE_t(
    const char* errstr,
    const void* private_info,
    size_t cb,
    void* user_data);

typedef clLogMessagesToSystemLogAPPLE_t *
clLogMessagesToSystemLogAPPLE_fn CL_API_SUFFIX__VERSION_1_0;

typedef void CL_API_CALL
clLogMessagesToStdoutAPPLE_t(
    const char* errstr,
    const void* private_info,
    size_t cb,
    void* user_data);

typedef clLogMessagesToStdoutAPPLE_t *
clLogMessagesToStdoutAPPLE_fn CL_API_SUFFIX__VERSION_1_0;

typedef void CL_API_CALL
clLogMessagesToStderrAPPLE_t(
    const char* errstr,
    const void* private_info,
    size_t cb,
    void* user_data);

typedef clLogMessagesToStderrAPPLE_t *
clLogMessagesToStderrAPPLE_fn CL_API_SUFFIX__VERSION_1_0;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY void CL_API_CALL
clLogMessagesToSystemLogAPPLE(
    const char* errstr,
    const void* private_info,
    size_t cb,
    void* user_data) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY void CL_API_CALL
clLogMessagesToStdoutAPPLE(
    const char* errstr,
    const void* private_info,
    size_t cb,
    void* user_data) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY void CL_API_CALL
clLogMessagesToStderrAPPLE(
    const char* errstr,
    const void* private_info,
    size_t cb,
    void* user_data) CL_API_SUFFIX__VERSION_1_0;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/***************************************************************
* cl_khr_icd
***************************************************************/
#define cl_khr_icd 1
#define CL_KHR_ICD_EXTENSION_NAME \
    "cl_khr_icd"


#define CL_KHR_ICD_EXTENSION_VERSION CL_MAKE_VERSION(2, 0, 0)

/* cl_platform_info */
#define CL_PLATFORM_ICD_SUFFIX_KHR                          0x0920

/* Error codes */
#define CL_PLATFORM_NOT_FOUND_KHR                           -1001

/* ICD 2 tag value */
#if INTPTR_MAX == INT32_MAX
#define CL_ICD2_TAG_KHR ((intptr_t)0x434C3331)
#else
#define CL_ICD2_TAG_KHR ((intptr_t)0x4F50454E434C3331)
#endif


typedef cl_int CL_API_CALL
clIcdGetPlatformIDsKHR_t(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms);

typedef clIcdGetPlatformIDsKHR_t *
clIcdGetPlatformIDsKHR_fn ;

typedef void* CL_API_CALL
clIcdGetFunctionAddressForPlatformKHR_t(
    cl_platform_id platform,
    const char* func_name);

typedef clIcdGetFunctionAddressForPlatformKHR_t *
clIcdGetFunctionAddressForPlatformKHR_fn ;

typedef cl_int CL_API_CALL
clIcdSetPlatformDispatchDataKHR_t(
    cl_platform_id platform,
    void* dispatch_data);

typedef clIcdSetPlatformDispatchDataKHR_t *
clIcdSetPlatformDispatchDataKHR_fn ;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms) ;

extern CL_API_ENTRY void* CL_API_CALL
clIcdGetFunctionAddressForPlatformKHR(
    cl_platform_id platform,
    const char* func_name) ;

extern CL_API_ENTRY cl_int CL_API_CALL
clIcdSetPlatformDispatchDataKHR(
    cl_platform_id platform,
    void* dispatch_data) ;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/***************************************************************
* cl_khr_il_program
***************************************************************/
#define cl_khr_il_program 1
#define CL_KHR_IL_PROGRAM_EXTENSION_NAME \
    "cl_khr_il_program"


#define CL_KHR_IL_PROGRAM_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

/* cl_device_info */
#define CL_DEVICE_IL_VERSION_KHR                            0x105B

/* cl_program_info */
#define CL_PROGRAM_IL_KHR                                   0x1169


typedef cl_program CL_API_CALL
clCreateProgramWithILKHR_t(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret);

typedef clCreateProgramWithILKHR_t *
clCreateProgramWithILKHR_fn CL_API_SUFFIX__VERSION_1_2;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithILKHR(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret) CL_API_SUFFIX__VERSION_1_2;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/***************************************************************
* cl_khr_image2d_from_buffer
***************************************************************/
#define cl_khr_image2d_from_buffer 1
#define CL_KHR_IMAGE2D_FROM_BUFFER_EXTENSION_NAME \
    "cl_khr_image2d_from_buffer"


#define CL_KHR_IMAGE2D_FROM_BUFFER_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

/* cl_device_info */
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR                 0x104A
#define CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR          0x104B

/***************************************************************
* cl_khr_initialize_memory
***************************************************************/
#define cl_khr_initialize_memory 1
#define CL_KHR_INITIALIZE_MEMORY_EXTENSION_NAME \
    "cl_khr_initialize_memory"


#define CL_KHR_INITIALIZE_MEMORY_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

typedef cl_bitfield         cl_context_memory_initialize_khr;

/* cl_context_properties */
#define CL_CONTEXT_MEMORY_INITIALIZE_KHR                    0x2030

/* cl_context_memory_initialize_khr */
#define CL_CONTEXT_MEMORY_INITIALIZE_LOCAL_KHR              (1 << 0)
#define CL_CONTEXT_MEMORY_INITIALIZE_PRIVATE_KHR            (1 << 1)

/***************************************************************
* cl_khr_terminate_context
***************************************************************/
#define cl_khr_terminate_context 1
#define CL_KHR_TERMINATE_CONTEXT_EXTENSION_NAME \
    "cl_khr_terminate_context"


#define CL_KHR_TERMINATE_CONTEXT_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

typedef cl_bitfield         cl_device_terminate_capability_khr;

/* cl_device_info */
#define CL_DEVICE_TERMINATE_CAPABILITY_KHR                  0x2031

/* cl_context_properties */
#define CL_CONTEXT_TERMINATE_KHR                            0x2032

/* cl_device_terminate_capability_khr */
#define CL_DEVICE_TERMINATE_CAPABILITY_CONTEXT_KHR          (1 << 0)

/* Error codes */
#define CL_CONTEXT_TERMINATED_KHR                           -1121


typedef cl_int CL_API_CALL
clTerminateContextKHR_t(
    cl_context context);

typedef clTerminateContextKHR_t *
clTerminateContextKHR_fn CL_API_SUFFIX__VERSION_1_2;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clTerminateContextKHR(
    cl_context context) CL_API_SUFFIX__VERSION_1_2;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/***************************************************************
* cl_khr_spir
***************************************************************/
#define cl_khr_spir 1
#define CL_KHR_SPIR_EXTENSION_NAME \
    "cl_khr_spir"


#define CL_KHR_SPIR_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

/* cl_device_info */
#define CL_DEVICE_SPIR_VERSIONS                             0x40E0

/* cl_program_binary_type */
#define CL_PROGRAM_BINARY_TYPE_INTERMEDIATE                 0x40E1

/***************************************************************
* cl_khr_create_command_queue
***************************************************************/
#define cl_khr_create_command_queue 1
#define CL_KHR_CREATE_COMMAND_QUEUE_EXTENSION_NAME \
    "cl_khr_create_command_queue"


#define CL_KHR_CREATE_COMMAND_QUEUE_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

typedef cl_properties       cl_queue_properties_khr;


typedef cl_command_queue CL_API_CALL
clCreateCommandQueueWithPropertiesKHR_t(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties_khr* properties,
    cl_int* errcode_ret);

typedef clCreateCommandQueueWithPropertiesKHR_t *
clCreateCommandQueueWithPropertiesKHR_fn CL_API_SUFFIX__VERSION_1_2;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueueWithPropertiesKHR(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties_khr* properties,
    cl_int* errcode_ret) CL_API_SUFFIX__VERSION_1_2;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

/***************************************************************
* cl_nv_device_attribute_query
***************************************************************/
#define cl_nv_device_attribute_query 1
#define CL_NV_DEVICE_ATTRIBUTE_QUERY_EXTENSION_NAME \
    "cl_nv_device_attribute_query"


#define CL_NV_DEVICE_ATTRIBUTE_QUERY_EXTENSION_VERSION CL_MAKE_VERSION(0, 0, 0)

/* cl_device_info */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV               0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV               0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV                    0x4002
#define CL_DEVICE_WARP_SIZE_NV                              0x4003
#define CL_DEVICE_GPU_OVERLAP_NV                            0x4004
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV                    0x4005
#define CL_DEVICE_INTEGRATED_MEMORY_NV                      0x4006

/***************************************************************
* cl_amd_device_attribute_query
***************************************************************/
#define cl_amd_device_attribute_query 1
#define CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXTENSION_NAME \
    "cl_amd_device_attribute_query"


#define CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXTENSION_VERSION CL_MAKE_VERSION(0, 0, 0)

/* cl_device_info */
#define CL_DEVICE_PROFILING_TIMER_OFFSET_AMD                0x4036
#define CL_DEVICE_TOPOLOGY_AMD                              0x4037
#define CL_DEVICE_BOARD_NAME_AMD                            0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD                    0x4039
#define CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD                 0x4040
#define CL_DEVICE_SIMD_WIDTH_AMD                            0x4041
#define CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD                0x4042
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD                       0x4043
#define CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD                   0x4044
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD              0x4045
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD         0x4046
#define CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD       0x4047
#define CL_DEVICE_LOCAL_MEM_BANKS_AMD                       0x4048
#define CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD                0x4049
#define CL_DEVICE_GFXIP_MAJOR_AMD                           0x404A
#define CL_DEVICE_GFXIP_MINOR_AMD                           0x404B
#define CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD                0x404C
#define CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD             0x4030
#define CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD                   0x4031
#define CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD        0x4033
#define CL_DEVICE_PCIE_ID_AMD                               0x4034

/***************************************************************
* cl_arm_printf
***************************************************************/
#define cl_arm_printf 1
#define CL_ARM_PRINTF_EXTENSION_NAME \
    "cl_arm_printf"


#define CL_ARM_PRINTF_EXTENSION_VERSION CL_MAKE_VERSION(0, 0, 0)

/* cl_context_properties */
#define CL_PRINTF_CALLBACK_ARM                              0x40B0
#define CL_PRINTF_BUFFERSIZE_ARM                            0x40B1

/***************************************************************
* cl_ext_device_fission
***************************************************************/
#define cl_ext_device_fission 1
#define CL_EXT_DEVICE_FISSION_EXTENSION_NAME \
    "cl_e