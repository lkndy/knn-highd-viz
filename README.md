# Hierarchical K-NN Graph Embedding

Implementation of "Visualizing large-scale high-dimensional data via hierarchical embedding of KNN graphs" (Zhu et al., 2021) in C++20.

## Features

- **Approximate K-NN construction** using EFANNA-style method (random projection trees + NN-Descent) per details.md Section 5
- **Multi-level graph coarsening** for hierarchical refinement
- **Cluster-based gradient sharing** for efficient optimization
- **LargeVis-style objective** with edge and negative sampling
- **Parallelized** using thread pools

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
- Python 3.11+ with [uv](https://github.com/astral-sh/uv) for Python scripts

## Building

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

### Quick Start with MNIST

```bash
# 1. Generate MNIST dataset (10,000 samples)
uv run python generate_test_data.py --N 10000 --output mnist_data

# 2. Run embedding
./build/embed_hknn \
  --input mnist_data.fvecs \
  --N 10000 \
  --D 784 \
  --K 100 \
  --perplexity 50 \
  --out mnist_embedding \
  --dim 2 \
  --seed 123

# 3. Visualize results
uv run python visualize.py mnist_embedding.csv mnist_embedding.png

# 4. Analyze embedding quality
uv run python analyze_embedding.py mnist_embedding.csv --data mnist_data.fvecs
```

### Basic Usage

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
- `--epochs_coarse`: Multiplier for coarse level epochs (default: 0 = use T_l = ceil(500 * |V^l|))
- `--epochs_fine`: Multiplier for finest level epochs (default: 0 = use T_l = ceil(500 * |V^l|))
- `--mneg`: Number of negative samples (default: 5)
- `--gamma`: Negative sampling balance for coarse levels (default: 7.0, finest uses 1.0)
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

## Python Scripts

All Python scripts should be run with `uv run python`:

- `generate_test_data.py`: Generate MNIST test data
  ```bash
  uv run python generate_test_data.py --N 10000 --output mnist_data
  ```

- `visualize.py`: Visualize embedding results
  ```bash
  uv run python visualize.py embedding.csv output.png
  ```

- `analyze_embedding.py`: Comprehensive embedding quality analysis
  ```bash
  uv run python analyze_embedding.py embedding.csv --data data.fvecs --k 30
  ```

## Testing

```bash
cd build
ctest
```

## Algorithm Overview

1. **K-NN Construction**: Build approximate K-NN graph using EFANNA-style method (random projection trees + NN-Descent refinement) per details.md Section 5
   - Time complexity: O(NK) per pass (not O(N²D))
   - Uses multiple random projection trees for initialization
   - NN-Descent refinement for iterative improvement
2. **Probability Computation**: Compute symmetric p_ij probabilities using perplexity-based binary search
3. **Hierarchical Coarsening**: Build multi-level graph pyramid via greedy grouping
4. **Hierarchical Refinement**: Optimize embeddings from coarse to fine levels using gradient sharing
   - Epochs per level: T_l = ceil(500 * |V^l|) as specified in details.md line 725
   - Gamma: γ=1 for finest level, γ=7 for coarse levels
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

