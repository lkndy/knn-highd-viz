#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef HKNN_USE_AVX512
#include <immintrin.h>
#elif defined(HKNN_USE_AVX2)
#include <immintrin.h>
#endif

namespace hknn {

// Stable squared L2 distance with epsilon clamping
inline float squared_distance(const float* __restrict a, 
                              const float* __restrict b, 
                              int dim) {
    float d2 = 0.0f;
    
#ifdef HKNN_USE_AVX512
    // AVX-512 implementation (process 16 floats at a time)
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(diff, diff);
        d2 += _mm512_reduce_add_ps(sq);
    }
    // Handle remaining elements
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        d2 += diff * diff;
    }
#elif defined(HKNN_USE_AVX2)
    // AVX2 implementation (process 8 floats at a time)
    int i = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    // Horizontal sum
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    for (int j = 0; j < 8; ++j) {
        d2 += temp[j];
    }
    // Handle remaining elements
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        d2 += diff * diff;
    }
#else
    // Scalar fallback
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        d2 += diff * diff;
    }
#endif
    
    return std::max(d2, 1e-12f);  // Clamp to avoid numerical issues
}

// Stable inverse kernel: 1 / (1 + d^2)
inline float inv_one_plus_sqdist(const float* __restrict a,
                                 const float* __restrict b, 
                                 int dim) {
    float d2 = squared_distance(a, b, dim);
    return 1.0f / (1.0f + d2);
}

// Compute squared distance for a batch (blocked for cache efficiency)
inline void squared_distances_blocked(const float* __restrict X,
                                      const float* __restrict Y,
                                      int N, int M, int D,
                                      float* __restrict distances) {
    constexpr int BLOCK_SIZE = 64;  // Cache-friendly block size
    
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int i_end = std::min(i + BLOCK_SIZE, N);
        for (int j = 0; j < M; j += BLOCK_SIZE) {
            int j_end = std::min(j + BLOCK_SIZE, M);
            
            // Compute distances in this block
            for (int ii = i; ii < i_end; ++ii) {
                for (int jj = j; jj < j_end; ++jj) {
                    distances[ii * M + jj] = squared_distance(
                        X + ii * D, Y + jj * D, D);
                }
            }
        }
    }
}

} // namespace hknn

