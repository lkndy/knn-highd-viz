#pragma once

#include "../math/kernels.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace hknn {
namespace embed {

// Positive gradient term: -2 * p_ij * (y_i - y_j) / (1 + ||y_i - y_j||^2)
inline void grad_pair_pos(float* __restrict gi,
                          const float* __restrict yi,
                          const float* __restrict yj,
                          float pij,
                          int dim) {
    float inv = inv_one_plus_sqdist(yi, yj, dim);
    for (int t = 0; t < dim; ++t) {
        gi[t] += -2.0f * pij * (yi[t] - yj[t]) * inv;
    }
}

// Negative gradient term: 2 * gamma * p_ij * (y_i - y_k) / (||y_i - y_k||^2 * (1 + ||y_i - y_k||^2))
inline void grad_pair_neg(float* __restrict gi,
                          const float* __restrict yi,
                          const float* __restrict yk,
                          float pij,
                          float gamma,
                          int dim) {
    // Compute squared distance
    float d2 = 0.0f;
    for (int t = 0; t < dim; ++t) {
        float d = yi[t] - yk[t];
        d2 += d * d;
    }
    d2 = std::max(d2, 1e-12f);
    
    float denom = d2 * (1.0f + d2);
    denom = std::max(denom, 1e-18f);
    
    float coeff = 2.0f * gamma * pij / denom;
    for (int t = 0; t < dim; ++t) {
        gi[t] += coeff * (yi[t] - yk[t]);
    }
}

// Clip gradient to avoid extreme updates
inline void clip_gradient(float* __restrict g, int dim, float max_norm = 10.0f) {
    float norm_sq = 0.0f;
    for (int t = 0; t < dim; ++t) {
        norm_sq += g[t] * g[t];
    }
    
    float norm = std::sqrt(norm_sq);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int t = 0; t < dim; ++t) {
            g[t] *= scale;
        }
    }
}

} // namespace embed
} // namespace hknn

