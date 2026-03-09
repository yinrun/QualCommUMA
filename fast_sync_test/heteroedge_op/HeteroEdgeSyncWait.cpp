//=============================================================================
//  HeteroEdge HTP Op Package - SyncWait implementation
//
//  DSP-side execution:
//  1. Get DSP VA for the flag ION buffer via HAP_mmap_get(ion_fd)
//  2. Poll the DSP VA directly (bypasses QNN TCM DMA copy)
//  3. Invalidate DSP data cache for input data buffer
//  4. Memcpy input data → output (establishes tensor dependency for RmsNorm)
//
//  Static parameter "flag_ion_fd" (UINT32 scalar):
//    The file descriptor of the flag ION buffer (rpcmem_to_fd result).
//    When > 0: HAP_mmap_get(fd) is used to get the DSP VA for direct DDR polling.
//    When 0:   fallback to QNN tensor pointer (TCM DMA copy — will timeout
//              if CPU/GPU writes after graphExecute starts).
//=============================================================================

#include <cstring>

// HAP_mmap_get is a Hexagon DSP-side API to map an ION fd to DSP VA.
// Only available on __hexagon__ target (DSP execution), not on ARM (graph prep).
#ifdef __hexagon__
#include "HAP_mem.h"
#endif

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_SyncWait);

// Static parameter: ION file descriptor of the flag buffer.
// Passed as a UINT32 scalar Qnn_Param_t named "flag_ion_fd".
// On DSP: HAP_mmap_get(fd) → DSP VA for direct DDR polling.
DEF_PACKAGE_PARAM_ORDER("SyncWait", "flag_ion_fd", false, nullptr)

// Forward declarations
template <typename Ttype>
int syncwait_impl(Ttype &out, const Ttype &data_in, const Ttype &flag_in,
                  const Tensor &flag_ion_fd);

template <typename DType>
int syncwait_fp16_impl(DType &out, const DType &data_in, const Tensor &flag_in,
                       const Tensor &flag_ion_fd);

// Generic fallback (used during graph compilation on ARM host)
DEF_PACKAGE_OP((syncwait_impl<Tensor>), "SyncWait")

// HVX-specialized variants
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((syncwait_fp16_impl<PlainFloat16Tensor>),
                                  "SyncWait",
                                  FAST,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((syncwait_fp16_impl<PlainFloat16Tensor_TCM>),
                                  "SyncWait",
                                  FAST,
                                  Flags::RESOURCE_HVX)

// Tensor layout: flat for data input/output; flag is unconstrained (UINT32 scalar)
DEF_TENSOR_PROPERTIES(Op("SyncWait", "data", "flag"), Flat("*", "data"))

//=============================================================================
// SyncWait implementation
//
// Input 0 (data):      FP16 tensor {1,1,1,hidden_dim} — GPU's output data
// Input 1 (flag):      UINT32 tensor {1,1,1,1}         — GPU done flag (QNN copy)
// Param  0 (flag_ion_fd): UINT32 scalar — ION fd for the flag buffer
// Output 0 (out):      FP16 tensor same dims as data    — copy of data input
//=============================================================================

// HVX-optimized variant: DType = PlainFloat16Tensor or PlainFloat16Tensor_TCM
template <typename DType>
int syncwait_fp16_impl(DType &out, const DType &data_in, const Tensor &flag_in,
                       const Tensor &flag_ion_fd) {
  auto [b, h, w, d] = data_in.dims();
  const size_t data_bytes = (size_t)b * h * w * d * 2;  // FP16 = 2 bytes/element

#ifdef __hexagon__
  // Read ION fd from static param (raw bytes to avoid float conversion)
  uint32_t ion_fd = 0;
  if (flag_ion_fd.raw_data_const())
    memcpy(&ion_fd, flag_ion_fd.raw_data_const(), sizeof(uint32_t));

  if (ion_fd > 0) {
    // ---- Path A: Direct DDR polling via HAP_mmap_get ----
    //
    // HAP_mmap_get(fd, &vaddr, &paddr) returns the DSP VA for the ION buffer.
    // This VA maps to the same physical DDR pages as the CPU's ION buffer.
    // Polling this VA with dcinva (cache invalidate) forces a DDR read,
    // bypassing QNN's TCM DMA copy of the flag tensor.
    //
    // Pre-condition: the ION fd must be mapped to CDSP VA space.
    // QNN's memRegister(QNN_MEM_TYPE_ION) establishes this mapping.
    void *vaddr = nullptr;
    uint64 paddr = 0;
    int ret = HAP_mmap_get((int)ion_fd, &vaddr, &paddr);
    if (ret == 0 && vaddr != nullptr) {
      volatile uint32_t *pflag = (volatile uint32_t *)vaddr;
      const int kTimeout = 10000000;
      int timeout = kTimeout;
      while (timeout-- > 0) {
        asm volatile("dcinva(%0)" : : "r"(pflag));
        asm volatile("" ::: "memory");
        if (*pflag == 1) break;
      }
      HAP_mmap_put((int)ion_fd);  // release reference count
    } else {
      // HAP_mmap_get failed: fallback to TCM copy
      volatile uint32_t *pflag = (volatile uint32_t *)flag_in.raw_data_const();
      const int kTimeout = 10000000;
      int timeout = kTimeout;
      while (timeout-- > 0) {
        asm volatile("dcinva(%0)" : : "r"(pflag));
        asm volatile("" ::: "memory");
        if (*pflag == 1) break;
      }
    }
  } else {
    // ---- Path B: Fallback to QNN tensor pointer (TCM DMA copy) ----
    // Only works if flag was already 1 before graphExecute (pre-set).
    volatile uint32_t *pflag = (volatile uint32_t *)flag_in.raw_data_const();
    const int kTimeout = 10000000;
    int timeout = kTimeout;
    while (timeout-- > 0) {
      asm volatile("dcinva(%0)" : : "r"(pflag));
      asm volatile("" ::: "memory");
      if (*pflag == 1) break;
    }
  }

  // Invalidate DSP cache for data buffer
  const char *dptr = (const char *)data_in.raw_data_const();
  for (size_t off = 0; off < data_bytes; off += 32) {
    asm volatile("dcinva(%0)" : : "r"(dptr + off));
  }
  asm volatile("" ::: "memory");
#endif  // __hexagon__

  // Passthrough: copy data to output
  out.set_dims(data_in);
  memcpy(out.raw_data(), data_in.raw_data_const(), data_bytes);
  return GraphStatus::Success;
}

// Generic fallback (ARM host for graph compilation; never runs on DSP at inference)
template <typename Ttype>
int syncwait_impl(Ttype &out, const Ttype &data_in, const Ttype &flag_in,
                  const Tensor &flag_ion_fd) {
  auto [b, h, w, d] = data_in.dims();
  const size_t data_bytes = (size_t)b * h * w * d * 2;

#ifdef __hexagon__
  uint32_t ion_fd = 0;
  if (flag_ion_fd.raw_data_const())
    memcpy(&ion_fd, flag_ion_fd.raw_data_const(), sizeof(uint32_t));

  if (ion_fd > 0) {
    void *vaddr = nullptr;
    uint64 paddr = 0;
    int ret = HAP_mmap_get((int)ion_fd, &vaddr, &paddr);
    if (ret == 0 && vaddr != nullptr) {
      volatile uint32_t *pflag = (volatile uint32_t *)vaddr;
      const int kTimeout = 10000000;
      int timeout = kTimeout;
      while (timeout-- > 0) {
        asm volatile("dcinva(%0)" : : "r"(pflag));
        asm volatile("" ::: "memory");
        if (*pflag == 1) break;
      }
      HAP_mmap_put((int)ion_fd);
    } else {
      volatile uint32_t *pflag = (volatile uint32_t *)flag_in.raw_data_const();
      const int kTimeout = 10000000;
      int timeout = kTimeout;
      while (timeout-- > 0) {
        asm volatile("dcinva(%0)" : : "r"(pflag));
        asm volatile("" ::: "memory");
        if (*pflag == 1) break;
      }
    }
  } else {
    volatile uint32_t *pflag = (volatile uint32_t *)flag_in.raw_data_const();
    const int kTimeout = 10000000;
    int timeout = kTimeout;
    while (timeout-- > 0) {
      asm volatile("dcinva(%0)" : : "r"(pflag));
      asm volatile("" ::: "memory");
      if (*pflag == 1) break;
    }
  }

  const char *dptr = (const char *)data_in.raw_data_const();
  for (size_t off = 0; off < data_bytes; off += 32) {
    asm volatile("dcinva(%0)" : : "r"(dptr + off));
  }
  asm volatile("" ::: "memory");
#endif  // __hexagon__

  out.set_dims(data_in);
  memcpy(out.raw_data(), data_in.raw_data_const(), data_bytes);
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_SyncWait);
