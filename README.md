# Hierarchical K-NN Graph Embedding

Implementation of "Visualizing large-scale high-dimensional data via hierarchical embedding of KNN graphs" (Zhu et al., 2021) in C++20.

## Features

- **Brute-force K-NN construction** with SIMD-optimized distance kernels
- **Multi-level graph coarsening** for hierarchical refinement
- **Cluster-based gradient sharing** for efficient optimization
- **LargeVis-style objective** with edge and negative sampling
- **Parallelized** using Boost.Asio thread pools

## Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.24 or higher
- Boost libraries:
  - program_options
  - filesystem
  - random
  - thread
  - system
  - timer
  - dynamic_bitset (header-only)

## Building

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
./embed_hknn \
  --input data.fvecs \
  --N 100000 \
  --D 128 \
  --K 100 \
  --perplexity 50 \
  --out output_embedding \
  --dim 2 \
  --seed 123
```

### Command-line Options

- `--input`: Input data file (.fvecs or raw float32)
- `--N`: Number of data points
- `--D`: Dimension of data points
- `--out`: Output embedding file (base path, generates .f32 and .csv)
- `--K`: Number of nearest neighbors (default: 100)
- `--perplexity`: Target perplexity for p_ij (default: 50.0)
- `--kml`: Grouping parameter for coarsening (default: 3)
- `--rho`: Minimum shrink ratio per level (default: 0.8)
- `--epochs_coarse`: Epochs per coarse level (default: 4)
- `--epochs_fine`: Epochs for finest level (default: 8)
- `--mneg`: Number of negative samples (default: 5)
- `--gamma`: Negative sampling balance (default: 5.0)
- `--lr`: Learning rate (default: 200.0)
- `--dim`: Embedding dimension, 2 or 3 (default: 2)
- `--seed`: Random seed (default: 123)
- `--threads`: Number of threads, 0 for auto (default: 0)

## File Formats

### Input

- **.fvecs**: FAISS format (int32 D header per row + D float32 values)
- **Raw float32**: Row-major layout, N×D floats

### Output

- **.f32**: Raw float32 row-major, N×dim
- **.csv**: CSV format with id,x,y[,z] columns

## Testing

```bash
cd build
ctest
```

## Algorithm Overview

1. **K-NN Construction**: Build K-nearest neighbor graph using brute-force with SIMD-optimized distance computation
2. **Probability Computation**: Compute symmetric p_ij probabilities using perplexity-based binary search
3. **Hierarchical Coarsening**: Build multi-level graph pyramid via greedy grouping
4. **Hierarchical Refinement**: Optimize embeddings from coarse to fine levels using gradient sharing
5. **Output**: Write 2D/3D embeddings to disk

## Performance

- Designed for datasets with N up to ~150k-250k points
- Supports dimensions D up to ~128 (higher dimensions may require NN-Descent)
- Multi-threaded K-NN construction and optimization
- SIMD-optimized distance kernels (AVX2/AVX-512 when available)

## License

This implementation is provided as-is for research and educational purposes.

## References

Zhu, X., et al. "Visualizing large-scale high-dimensional data via hierarchical embedding of KNN graphs." (2021)

