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
from tilelang.utils.target import determine_target, target_get_warp_size


def _has_vulkan_codegen() -> bool:
    return tvm.get_global_func("target.build.vulkan", allow_missing=True) is not None


requires_vulkan_codegen = pytest.mark.skipif(
    not _has_vulkan_codegen(),
    reason="Requires TVM Vulkan codegen support (`target.build.vulkan`)",
)

_VK_API_VERSION_1_1 = (1 << 22) | (1 << 12)
_SPIRV_VERSION_1_3 = 0x10300


def make_test_vulkan_target(device: str | None = None, **overrides) -> tvm.target.Target:
    attrs = {
        "kind": "vulkan",
        "supports_storage_buffer_storage_class": True,
        "vulkan_api_version": _VK_API_VERSION_1_1,
        "max_spirv_version": _SPIRV_VERSION_1_3,
        "supports_float32": True,
        "supports_int32": True,
        "max_threads_per_block": 256,
        "max_num_threads": 256,
        "max_shared_memory_per_block": 32768,
        "thread_warp_size": 1,
    }
    if device:
        attrs["device"] = device
    attrs.update(overrides)
    return tvm.target.Target(attrs)


def make_test_adreno_target(**overrides) -> tvm.target.Target:
    attrs = {
        "supports_float16": True,
        "supports_16bit_buffer": True,
        "max_shared_memory_per_block": 32768,
    }
    attrs.update(overrides)
    return make_test_vulkan_target(device="adreno", **attrs)


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


def elemwise_sub(N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N, threads=1) as (bx,):
            C[bx] = A[bx] - B[bx]

    return main


def elemwise_fma(N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        D: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N, threads=1) as (bx,):
            D[bx] = A[bx] * B[bx] + C[bx]

    return main


def matrix_add(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = (
                    A[by * block_M + i, bx * block_N + j] + B[by * block_M + i, bx * block_N + j]
                )

    return main


def copy_2d(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            T.copy(
                A[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N],
                B[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N],
            )

    return main


def fragment_fill_and_copy(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def main(
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def copy_through_shared(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared")
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def clear_shared(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def main(
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            buf = T.alloc_shared((block_M, block_N), dtype, scope="shared")
            T.clear(buf)
            T.copy(buf, C[by * block_M, bx * block_N])

    return main


def reduce_sum_shared(M, N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, B_shared, dim=1)
            T.copy(B_shared, B)

    return main


def reduce_max_shared(M, N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)
            T.copy(A, A_shared)
            T.reduce_max(A_shared, B_shared, dim=1)
            T.copy(B_shared, B)

    return main


def reduce_sum_fragment(M, N, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)
            T.copy(A, A_local)
            T.reduce_sum(A_local, B_local, dim=1)
            T.copy(B_local, B)

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


def simple_serial_loop(N, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as (bx,):
            for i in T.Serial(block_N):
                B[bx * block_N + i] = A[bx * block_N + i] * 2.0

    return main


def nested_serial_loops(M, N, block_M, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i in T.Serial(block_M):
                for j in T.Serial(block_N):
                    B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j] + 1.0

    return main


def parallel_serial_combo(M, N, block_M, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                for k in T.Serial(4):
                    B[by * block_M + i, bx * block_N + j] += A[by * block_M + i, bx * block_N + j]

    return main


def matmul_serial_only(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float32),
        B: T.Tensor((K, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float32, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), T.float32, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for ko in T.Serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, k] * B_shared[k, j]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def assert_vulkan_codegen(func):
    target = make_test_vulkan_target()
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)

    src_code = artifact.kernel_source
    _maybe_print_kernel_source(src_code)
    assert src_code is not None
    assert len(src_code) > 0

@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_add():
    assert_vulkan_codegen(elemwise_add(1024))

@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_sub():
    assert_vulkan_codegen(elemwise_sub(1024))

@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_mul_float32():
    assert_vulkan_codegen(elemwise_mul(1024, dtype=T.float32))

@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_mul_int32():
    assert_vulkan_codegen(elemwise_mul(1024, dtype=T.int32))

@requires_vulkan_codegen
def test_vulkan_codegen_elemwise_fma():
    assert_vulkan_codegen(elemwise_fma(1024))

@requires_vulkan_codegen
def test_vulkan_codegen_matrix_add():
    assert_vulkan_codegen(matrix_add(256, 256, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_copy_2d():
    assert_vulkan_codegen(copy_2d(256, 256, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_fragment_fill():
    assert_vulkan_codegen(fragment_fill_and_copy(256, 256, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_copy_through_shared():
    assert_vulkan_codegen(copy_through_shared(256, 256, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_matmul():
    assert_vulkan_codegen(matmul(1024, 1024, 1024, 16, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_simple_serial_loop():
    assert_vulkan_codegen(simple_serial_loop(1024, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_nested_serial_loops():
    assert_vulkan_codegen(nested_serial_loops(256, 256, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_parallel_serial_combo():
    assert_vulkan_codegen(parallel_serial_combo(256, 256, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_matmul_serial_only():
    assert_vulkan_codegen(matmul_serial_only(256, 256, 256, 16, 16, 16))

@requires_vulkan_codegen
def test_vulkan_codegen_clear_shared():
    assert_vulkan_codegen(clear_shared(256, 256, 16, 16))

@requires_vulkan_codegen
@pytest.mark.skip(
    reason="T.reduce emits tl::AllReduce extern call, which SPIR-V rejects."
)
def test_vulkan_codegen_reduce_sum_shared():
    assert_vulkan_codegen(reduce_sum_shared(64, 64))

@requires_vulkan_codegen
@pytest.mark.skip(
    reason="T.reduce emits tl::AllReduce extern call, which SPIR-V rejects."
)
def test_vulkan_codegen_reduce_max_shared():
    assert_vulkan_codegen(reduce_max_shared(64, 64))

@requires_vulkan_codegen
@pytest.mark.skip(
    reason="T.reduce emits tl::AllReduce extern call, which SPIR-V rejects."
)
def test_vulkan_codegen_reduce_sum_fragment():
    assert_vulkan_codegen(reduce_sum_fragment(64, 64))

def test_vulkan_compile_rejected():
    with pytest.raises(ValueError):
        tilelang.compile(elemwise_add(16), out_idx=[2], target="vulkan")


@requires_vulkan_codegen
def test_vulkan_test_target_attrs():
    target = make_test_vulkan_target()
    assert target.kind.name == "vulkan"
    assert target.attrs.get("supports_storage_buffer_storage_class", None) is True
    assert target.attrs.get("vulkan_api_version", None) == (1 << 22) | (1 << 12)


@requires_vulkan_codegen
def test_vulkan_test_adreno_target_attrs():
    """Should set Adreno-specific attrs for explicit test targets."""
    target = make_test_adreno_target()
    assert target.kind.name == "vulkan"
    assert "adreno" in target.keys
    assert target.attrs.get("supports_float16", None) is True
    assert target.attrs.get("supports_16bit_buffer", None) is True


@requires_vulkan_codegen
def test_vulkan_codegen_uses_storage_buffer():
    """Should produce StorageBuffer, not Uniform+BufferBlock"""
    target = make_test_vulkan_target()
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(elemwise_add(256), target=target)
    src = artifact.kernel_source
    _maybe_print_kernel_source(src)
    assert src is not None
    assert len(src) > 0
    assert "BufferBlock" not in src


if __name__ == "__main__":
    tilelang.testing.main()
