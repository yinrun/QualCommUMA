# ===- rmsnorm_kernel.py ----------------------------------------------------===
#
# Triton RMSNorm kernel for Hexagon HTP via hexagon-mlir.
# Benchmark harness matching custom_op/test output format.
#
# Usage:
#   pytest -sv rmsnorm_kernel.py                    # run all
#   pytest -sv rmsnorm_kernel.py -k "decode"        # decode only
#   pytest -sv rmsnorm_kernel.py -k "fp16"          # FP16 only
#
# ===------------------------------------------------------------------------===

import time
import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

EPSILON = 1e-5

# ---------------------------------------------------------------------------
# Test scenarios matching custom_op/test benchmark
# ---------------------------------------------------------------------------
SCENARIOS = [
    # (name,      batch, hidden)
    ("decode-2k",     1, 2048),
    ("decode-3.2k",   1, 3200),
    ("decode-4k",     1, 4096),
    ("prefill-16",   16, 4096),
    ("prefill-64",   64, 4096),
    ("pf-256",      256, 4096),
    ("pf-512",      512, 4096),
    ("pf-1k",      1024, 4096),
]

# Optimization option combinations to sweep
OPT_CONFIGS = [
    # (label, num_threads, enableMultiThreading, enableVTCMTiling, enableHexagonmemCopyToDMA)
    ("1T-base",  1, True,  True,  True),
    ("4T-base",  4, True,  True,  True),
    ("4T-noDMA", 4, True,  True,  False),
    ("4T-noVT",  4, True,  False, True),
    ("4T-bare",  4, False, False, False),
]


# ---------------------------------------------------------------------------
# Kernel definition helpers
# ---------------------------------------------------------------------------
# hexagon-mlir requires unique kernel function names per (dtype) combination
# because it compiles each variant to a separate Hexagon binary.
# We define the kernel inside the test function using @parameterize_func_name.

def _dtype_suffix(dtype):
    return {torch.float16: "float16", torch.float32: "float32"}[dtype]


def _parameterize_func_name(*parameters):
    """Rename JIT kernel to include dtype — required by hexagon-mlir."""
    def decorator(func):
        name = func.__name__
        for p in parameters:
            if isinstance(p, torch.dtype):
                name += "_" + _dtype_suffix(p)
        func.__name__ = name
        return func
    return decorator


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def rmsnorm_ref(x, weight, eps=EPSILON):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


# ---------------------------------------------------------------------------
# Correctness + latency test
# ---------------------------------------------------------------------------
class TestInfo:
    """Accumulates results for summary table."""
    rows = []

    @classmethod
    def add(cls, scene, batch, hidden, dtype, opt_label,
            latency_us, bw_gbs, max_err, passed):
        cls.rows.append((scene, batch, hidden, _dtype_suffix(dtype),
                         opt_label, latency_us, bw_gbs, max_err, passed))

    @classmethod
    def print_summary(cls):
        if not cls.rows:
            return
        hdr = (f"{'Scene':<14} {'batch':>5} {'hid':>5} {'dtype':<7} "
               f"{'opts':<10} | {'lat(us)':>9} {'BW(GB/s)':>9} "
               f"{'MaxErr':>12} {'OK':>3}")
        sep = "-" * len(hdr)
        print(f"\n{sep}\n  Triton RMSNorm Benchmark Summary\n{sep}")
        print(hdr)
        print(sep)
        for r in cls.rows:
            (scene, batch, hidden, dt, opt, lat, bw, err, ok) = r
            print(f"{scene:<14} {batch:>5} {hidden:>5} {dt:<7} "
                  f"{opt:<10} | {lat:>9.1f} {bw:>9.2f} "
                  f"{err:>12.6e} {'Y' if ok else 'N':>3}")
        print(sep)


# ---------------------------------------------------------------------------
# Parametrized benchmark tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def print_summary_at_end():
    yield
    TestInfo.print_summary()


@pytest.mark.parametrize(
    "scenario", SCENARIOS, ids=[s[0] for s in SCENARIOS]
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32],
    ids=["fp16", "fp32"]
)
@pytest.mark.parametrize(
    "opt_cfg", OPT_CONFIGS, ids=[c[0] for c in OPT_CONFIGS]
)
def test_rmsnorm(scenario, dtype, opt_cfg):
    scene_name, num_rows, num_cols = scenario
    opt_label, num_threads, mt, vtcm, dma = opt_cfg
    block_size = triton.next_power_of_2(num_cols)

    # Tolerance: FP16 needs more slack
    atol = 1e-2 if dtype == torch.float16 else 1e-5

    # ---- define kernel (must be inside test for unique name per dtype) ----
    @triton.jit
    @_parameterize_func_name(dtype)
    def rms_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        weights_ptr,
        BLOCK_SIZE: tl.constexpr,
        EPSILON: tl.constexpr,
        NUM_COLS: tl.constexpr,
        NUM_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        programs = tl.num_programs(0)
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < NUM_COLS
        for row in range(pid, NUM_ROWS, programs):
            x_ptr_incr = x_ptr + row * NUM_COLS
            y_ptr_incr = y_ptr + row * NUM_COLS

            x = tl.load(x_ptr_incr + block, mask=mask, other=0.0)
            mean_squared = tl.sum(x * x, axis=0) / NUM_COLS
            rms = tl.sqrt(mean_squared + EPSILON)

            g = tl.load(weights_ptr + block, mask=mask, other=0.0)
            y = (x / rms) * g
            tl.store(y_ptr_incr + block, y, mask=mask)

    # ---- allocate tensors ----
    x = torch.rand(num_rows, num_cols, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)
    weights = torch.rand(num_cols, dtype=dtype)

    # ---- warmup ----
    grid = (num_threads,)
    for _ in range(3):
        rms_norm_fwd_kernel[grid](
            x, output, weights,
            EPSILON=EPSILON,
            NUM_ROWS=num_rows,
            NUM_COLS=num_cols,
            BLOCK_SIZE=block_size,
            enableMultiThreading=mt,
            enableVTCMTiling=vtcm,
            enableConvertToHexagonmem=mt,
            enableHexagonmemCopyToDMA=dma,
        )

    # ---- timed runs ----
    n_iters = 20
    t0 = time.perf_counter()
    for _ in range(n_iters):
        rms_norm_fwd_kernel[grid](
            x, output, weights,
            EPSILON=EPSILON,
            NUM_ROWS=num_rows,
            NUM_COLS=num_cols,
            BLOCK_SIZE=block_size,
            enableMultiThreading=mt,
            enableVTCMTiling=vtcm,
            enableConvertToHexagonmem=mt,
            enableHexagonmemCopyToDMA=dma,
        )
    t1 = time.perf_counter()
    latency_us = (t1 - t0) / n_iters * 1e6

    # ---- correctness ----
    reference = rmsnorm_ref(x.float(), weights.float()).to(dtype)
    max_err = (output - reference).abs().max().item()
    passed = torch.allclose(output, reference, atol=atol)

    # ---- bandwidth: read x + weights, write y ----
    elem_bytes = 2 if dtype == torch.float16 else 4
    data_bytes = (num_rows * num_cols * 2 + num_cols) * elem_bytes
    bw_gbs = data_bytes / (latency_us * 1e-6) / 1e9

    TestInfo.add(scene_name, num_rows, num_cols, dtype, opt_label,
                 latency_us, bw_gbs, max_err, passed)

    print(f"  {scene_name:14s} b={num_rows:4d} h={num_cols:4d} "
          f"{_dtype_suffix(dtype):6s} {opt_label:10s} | "
          f"{latency_us:8.1f} us  {bw_gbs:6.2f} GB/s  "
          f"err={max_err:.6e}  {'PASS' if passed else 'FAIL'}")

    assert passed, (
        f"Correctness check failed: max_err={max_err:.6e} > atol={atol}"
    )
