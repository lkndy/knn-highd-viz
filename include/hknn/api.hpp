#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace hknn {

struct Embedding {
    std::vector<float> data; // Raw data (row-major)
    std::size_t N;
    std::size_t dim;
};

// High-level embedding call
Embedding run_hierarchy_embedding(
    const float* X, std::size_t N, std::size_t D,
    std::size_t K = 100,
    std::size_t target_dim = 2,
    float perplexity = 50.0f,
    std::size_t max_levels = 0, // auto (0 means auto-determine)
    int M = 5,
    float gamma_coarse = 7.0f,
    float gamma_fine = 1.0f,
    float base_lr = 200.0f,
    std::uint64_t seed = 12345,
    // Additional implementation-specific parameters
    std::uint32_t num_threads = 0
);

} // namespace hknn

