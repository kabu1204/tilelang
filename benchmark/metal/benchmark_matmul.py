import argparse
import csv
import gc
import time
from dataclasses import dataclass

import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_kernel(M, N, K, block_M, block_N, block_K, dtype=T.float32, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
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

    return gemm


@dataclass
class BenchmarkResult:
    dtype: str
    shape: tuple[int, int, int]
    best_config: tuple[int, int, int] | None
    max_diff: float | None
    tilelang_ms: float | None
    tilelang_gflops: float | None
    torch_ms: float
    torch_gflops: float


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Invalid shape '{text}', expected MxNxK")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Invalid shape '{text}', expected integers") from err
    return m, n, k


def parse_shapes(text: str) -> list[tuple[int, int, int]]:
    return [parse_shape(item) for item in text.split(",") if item.strip()]


def parse_configs(text: str) -> list[tuple[int, int, int]]:
    return [parse_shape(item) for item in text.split(",") if item.strip()]


def parse_dtypes(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_dtype(dtype_name: str) -> tuple[T.dtype, torch.dtype, float]:
    if dtype_name == "float16":
        return T.float16, torch.float16, 1.0
    if dtype_name == "float32":
        return T.float32, torch.float32, 1e-5
    if dtype_name == "int32":
        return T.int32, torch.int32, 0.0
    raise ValueError(f"Unsupported dtype '{dtype_name}'")


def sync_mps() -> None:
    torch.mps.synchronize()


def benchmark_callable(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    sync_mps()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync_mps()
    return (time.perf_counter() - start) * 1000 / repeat


def make_inputs(M: int, N: int, K: int, torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if torch_dtype == torch.int32:
        a = torch.randint(100, (M, K), dtype=torch_dtype, device="mps")
        b = torch.randint(100, (K, N), dtype=torch_dtype, device="mps")
    else:
        a = torch.randn(M, K, dtype=torch_dtype, device="mps")
        b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    return a, b


def ref_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.mm(a, b, out=out)


def run_case(
    M: int,
    N: int,
    K: int,
    dtype_name: str,
    configs: list[tuple[int, int, int]],
    warmup: int,
    repeat: int,
) -> BenchmarkResult:
    tl_dtype, torch_dtype, diff_limit = resolve_dtype(dtype_name)
    torch.manual_seed(0)
    a, b = make_inputs(M, N, K, torch_dtype)
    ref = torch.empty(M, N, dtype=torch_dtype, device="mps")
    ref_matmul(a, b, out=ref)

    best_ms = None
    best_cfg = None
    best_diff = None

    for block_M, block_N, block_K in configs:
        c = torch.zeros(M, N, dtype=torch_dtype, device="mps")
        try:
            kernel = matmul_kernel(M, N, K, block_M, block_N, block_K, dtype=tl_dtype, accum_dtype=T.float32)
            kernel(a, b, c)
            sync_mps()

            max_diff = (c - ref).abs().max().item()
            if max_diff > diff_limit:
                print(
                    f"shape={M}x{N}x{K} dtype={dtype_name} cfg={block_M}x{block_N}x{block_K} "
                    f"status=bad_result max_diff={max_diff}"
                )
                continue

            tilelang_ms = benchmark_callable(lambda: kernel(a, b, c), warmup=warmup, repeat=repeat)
            if best_ms is None or tilelang_ms < best_ms:
                best_ms = tilelang_ms
                best_cfg = (block_M, block_N, block_K)
                best_diff = max_diff
        except Exception as err:
            print(
                f"shape={M}x{N}x{K} dtype={dtype_name} cfg={block_M}x{block_N}x{block_K} "
                f"status=error err={type(err).__name__}: {err}"
            )
        finally:
            del c
            gc.collect()
            torch.mps.empty_cache()

    torch_out = torch.empty_like(ref)
    torch_ms = benchmark_callable(lambda: ref_matmul(a, b, out=torch_out), warmup=warmup, repeat=repeat)
    flops = 2 * M * N * K

    result = BenchmarkResult(
        dtype=dtype_name,
        shape=(M, N, K),
        best_config=best_cfg,
        max_diff=best_diff,
        tilelang_ms=best_ms,
        tilelang_gflops=None if best_ms is None else flops / (best_ms / 1000.0) / 1e9,
        torch_ms=torch_ms,
        torch_gflops=flops / (torch_ms / 1000.0) / 1e9,
    )

    del a, b, ref, torch_out
    gc.collect()
    torch.mps.empty_cache()
    return result


def print_result(result: BenchmarkResult) -> None:
    M, N, K = result.shape
    if result.best_config is None:
        print(
            f"shape={M}x{N}x{K} dtype={result.dtype} best_cfg=none "
            f"tilelang_ms=NA tilelang_gflops=NA "
            f"torch_ms={result.torch_ms:.3f} torch_gflops={result.torch_gflops:.1f}"
        )
        return

    block_M, block_N, block_K = result.best_config
    print(
        f"shape={M}x{N}x{K} dtype={result.dtype} "
        f"best_cfg={block_M}x{block_N}x{block_K} "
        f"max_diff={result.max_diff} "
        f"tilelang_ms={result.tilelang_ms:.3f} tilelang_gflops={result.tilelang_gflops:.1f} "
        f"torch_ms={result.torch_ms:.3f} torch_gflops={result.torch_gflops:.1f}"
    )


def write_csv(path: str, results: list[BenchmarkResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "dtype",
                "M",
                "N",
                "K",
                "best_block_M",
                "best_block_N",
                "best_block_K",
                "max_diff",
                "tilelang_ms",
                "tilelang_gflops",
                "torch_ms",
                "torch_gflops",
            ]
        )
        for result in results:
            block_M, block_N, block_K = result.best_config or ("", "", "")
            M, N, K = result.shape
            writer.writerow(
                [
                    result.dtype,
                    M,
                    N,
                    K,
                    block_M,
                    block_N,
                    block_K,
                    result.max_diff,
                    result.tilelang_ms,
                    result.tilelang_gflops,
                    result.torch_ms,
                    result.torch_gflops,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Metal matmul benchmark sweep")
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes("512x512x512,1024x1024x1024,2048x2048x2048"),
        help="Comma-separated shapes in MxNxK form",
    )
    parser.add_argument(
        "--dtypes",
        type=parse_dtypes,
        default=parse_dtypes("float16,float32"),
        help="Comma-separated dtype list",
    )
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=parse_configs("16x16x16,32x32x16,32x32x32,64x64x16"),
        help="Comma-separated tile configs in block_Mxblock_Nxblock_K form",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per measurement")
    parser.add_argument("--repeat", type=int, default=10, help="Timed iterations per measurement")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in the current Python runtime")

    print(f"torch {torch.__version__}")
    print(f"mps_available {torch.backends.mps.is_available()}")
    print("Benchmarking Metal matmul sweep...")
    print("")

    results = []
    for dtype_name in args.dtypes:
        print(f"=== dtype={dtype_name} ===")
        for M, N, K in args.shapes:
            result = run_case(M, N, K, dtype_name, args.configs, args.warmup, args.repeat)
            results.append(result)
            print_result(result)
        print("")

    if args.csv is not None:
        write_csv(args.csv, results)
        print(f"Wrote results to {args.csv}")


if __name__ == "__main__":
    main()
