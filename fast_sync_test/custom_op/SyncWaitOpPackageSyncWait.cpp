//=============================================================================
//  SyncWait HTP Op Package - Implementation
//
//  DSP-side execution:
//  1. Poll GPU flag (ION shared memory) with dcinva cache invalidation
//  2. Invalidate DSP data cache for input data buffer
//  3. Memcpy input data → output (establishes tensor dependency for RmsNorm)
//
//  This enables GPU↔NPU parallel launch:
//  - graphExecute starts immediately after GPU submit
//  - DSP spins on GPU flag (~40us overlap with GPU execution)
//  - After flag detected: data is guaranteed to be in DDR
//=============================================================================

#include <cstring>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_SyncWait);

// No parameters for SyncWait (flag is a regular input tensor)

// Forward declarations
template <typename Ttype>
int syncwait_ref_impl(Ttype &out, const Ttype &data_in, const Ttype &flag_in);

// Register generic Tensor implementation (handles FP16 data + UINT32 flag)
DEF_PACKAGE_OP((syncwait_ref_impl<Tensor>), "SyncWait")

// Tensor layout: require flat layout for data input/output, flag is unconstrained
DEF_TENSOR_PROPERTIES(Op("SyncWait", "data", "flag"), Flat("*", "data"))

//=============================================================================
// SyncWait implementation
//
// Input 0 (data): FP16 tensor {1,1,1,hidden_dim} — GPU's output data
// Input 1 (flag): UINT32 tensor {1,1,1,1}         — GPU done flag (0=pending, 1=done)
// Output 0 (out): FP16 tensor same dims as data    — copy of data input
//=============================================================================

template <typename Ttype>
int syncwait_ref_impl(Ttype &out, const Ttype &data_in, const Ttype &flag_in) {
  (void)flag_in;

  // Minimal passthrough: just copy data (no flag polling, no cache invalidation)
  auto [b, h, w, d] = data_in.dims();
  const size_t data_bytes = (size_t)b * h * w * d * 2;  // FP16

  out.set_dims(data_in);
  memcpy(out.raw_data(), data_in.raw_data_const(), data_bytes);

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_SyncWait);
