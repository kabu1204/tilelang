"""Test Vulkan (SPIR-V) code generation.

These tests verify that TileLang can compile kernels down to SPIR-V
through TVM's Vulkan codegen without requiring a Vulkan runtime device.
"""

import os

import faulthandler
import pytest
import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def _has_vulkan_codegen() -> bool:
    return tvm.get_global_func("target.build.vulkan", allow_missing=True) is not None


requires_vulkan_codegen = pytest.mark.skipif(
    not _has_vulkan_codegen(),
    reason="Requires TVM Vulkan codegen support (`target.build.vulkan`)",
)


def _maybe_print_kernel_source(src_code: str) -> None:
    """Print kernel source when explicitly enabled.

    Pytest captures stdout by default; use `pytest -s` to see it live, or let it print on failures.
    """

    flag = os.getenv("TILELANG_VULKAN_PRINT_SRC", "")
    if not flag:
        return

    flag_norm = flag.lower()
    if flag_norm in {"1", "true", "yes", "on"}:
        print("\n=== TileLang Vulkan kernel source (begin) ===\n", flush=True)
        print(src_code, flush=True)
        print("\n=== TileLang Vulkan kernel source (end) ===\n", flush=True)
        return

    # If the var isn't a boolean-like value, treat it as a file path to dump into.
    with open(flag, "w", encoding="utf-8") as f:
        f.write(src_code)
        if not src_code.endswith("\n"):
            f.write("\n")


if os.getenv("TILELANG_FAULTHANDLER", "").lower() in {"1", "true", "yes", "on"}:
    faulthandler.enable(all_threads=True)


def elemwise_add(N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        C: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N, threads=1) as (bx,):
            C[bx] = A[bx] + B[bx]

    return main


def elemwise_mul(N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N, threads=1) as (bx,):
            C[bx] = A[bx] * B[bx]

    return main


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float32, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def assert_vulkan_codegen(func):
    with tvm.transform.PassContext(), tvm.target.Target("vulkan"):
        artifact = tilelang.lower(func, target="vulkan")

    src_code = artifact.kernel_source
    _maybe_print_kernel_source(src_code)
    assert src_code is not None
    assert len(src_code) > 0


@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_add():
    assert_vulkan_codegen(elemwise_add(1024))


@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_mul_float32():
    assert_vulkan_codegen(elemwise_mul(1024, dtype=T.float32))


@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_mul_int32():
    assert_vulkan_codegen(elemwise_mul(1024, dtype=T.int32))


@pytest.mark.xfail(
    reason="TVM Vulkan/SPIR-V codegen emits invalid OpPhi for looped matmul kernels",
    raises=tvm.error.InternalError,
)
@requires_vulkan_codegen
def test_vulkan_codegen_matmul():
    assert_vulkan_codegen(matmul(1024, 1024, 1024, 16, 16, 16))


def test_vulkan_compile_rejected():
    with pytest.raises(ValueError):
        tilelang.compile(elemwise_add(16), out_idx=[2], target="vulkan")


if __name__ == "__main__":
    tilelang.testing.main()
