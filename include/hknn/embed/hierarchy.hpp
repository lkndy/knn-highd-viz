#pragma once

#include "optimizer.hpp"
#include "../graph/coarsen.hpp"
#include "../math/rng.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>

namespace hknn {
namespace embed {

// Initialize embedding with Gaussian noise
void random_init_embedding(std::vector<float>& Y,
                          uint32_t N,
                          int dim,
                          float sigma,
                          uint64_t seed) {
    Y.resize(N * dim);
    Rng rng(seed);
    
    for (uint32_t i = 0; i < N; ++i) {
        for (int d = 0; d < dim; ++d) {
            Y[i * dim + d] = RngManager::normal(rng, 0.0f, sigma);
        }
    }
}

// Prolongate embedding from coarse to fine level
void prolongate_embedding(const graph::Level& coarse_level,
                         graph::Level& fine_level,
                         int dim) {
    uint32_t N_fine = fine_level.num_vertices();
    fine_level.Y.resize(N_fine * dim);
    
    // Check if coarse level has embedding
    if (coarse_level.Y.empty() || coarse_level.num_vertices() == 0) {
        // Initialize fine level with small random noise
        Rng rng(12345);
        for (uint32_t v = 0; v < N_fine; ++v) {
            for (int d = 0; d < dim; ++d) {
                fine_level.Y[v * dim + d] = RngManager::normal(rng, 0.0f, 0.01f);
            }
        }
        return;
    }
    
    // Each fine vertex inherits its parent's coordinate
    for (uint32_t v = 0; v < N_fine; ++v) {
        uint32_t parent = fine_level.parent[v];
        if (parent != UINT32_MAX && parent < coarse_level.num_vertices()) {
            // Copy parent's coordinates
            for (int d = 0; d < dim; ++d) {
                fine_level.Y[v * dim + d] = coarse_level.Y[parent * dim + d];
            }
        } else {
            // No parent (shouldn't happen, but initialize with small noise)
            Rng rng(v + 12345);
            for (int d = 0; d < dim; ++d) {
                fine_level.Y[v * dim + d] = RngManager::normal(rng, 0.0f, 0.01f);
            }
        }
    }
}

// Hierarchical refinement: coarse to fine
void refine_hierarchical(std::vector<graph::Level>& hierarchy,
                        const OptimConfig& base_cfg,
                        uint32_t epochs_coarse,
                        uint32_t epochs_fine,
                        uint32_t num_threads,
                        uint64_t seed) {
    
    if (hierarchy.empty()) {
        return;
    }
    
    int num_levels = static_cast<int>(hierarchy.size());
    int dim = base_cfg.dim;
    
    // Refine from coarse to fine
    // hierarchy[0] = finest, hierarchy[num_levels-1] = coarsest
    // We iterate from coarsest (num_levels-1) to finest (0)
    for (int level_idx = num_levels - 1; level_idx >= 0; --level_idx) {
        graph::Level& level = hierarchy[level_idx];
        
        if (level_idx == num_levels - 1) {
            // Coarsest level: initialize with random noise
            random_init_embedding(level.Y, level.num_vertices(), dim, 0.0001f, seed);
        } else {
            // Fine level: prolongate from the next coarser level (level_idx+1)
            // level_idx+1 is coarser than level_idx
            prolongate_embedding(hierarchy[level_idx + 1], level, dim);
        }
        
        // Determine epochs for this level
        uint32_t epochs = (level_idx == 0) ? epochs_fine : epochs_coarse;
        
        // Adjust gamma for this level (higher on coarse, lower on fine)
        OptimConfig level_cfg = base_cfg;
        if (num_levels > 1) {
            float gamma_scale = 1.0f + static_cast<float>(num_levels - 1 - level_idx) * 0.5f;
            level_cfg.gamma = base_cfg.gamma * gamma_scale;
        }
        
        // Optimize this level
        optimize_level(level, level_cfg, epochs, num_threads, seed + level_idx);
    }
}

} // namespace embed
} // namespace hknn

