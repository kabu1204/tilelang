# Metal Matmul Benchmark

This directory contains a reusable Apple Metal benchmark for the file-backed
TileLang matmul kernel used in the Metal test suite.

## Requirements

- macOS on Apple Silicon
- PyTorch with MPS support
- `torch.backends.mps.is_available()` must be `True`

## Usage

Run the default sweep:

```bash
python benchmark/metal/benchmark_matmul.py
```

Customize shapes, dtypes, and tile sizes:

```bash
python benchmark/metal/benchmark_matmul.py \
  --shapes 512x512x512,1024x1024x1024 \
  --dtypes float16,float32 \
  --configs 16x16x16,32x32x16,32x32x32
```

Write the results to CSV:

```bash
python benchmark/metal/benchmark_matmul.py --csv metal_matmul.csv
```
