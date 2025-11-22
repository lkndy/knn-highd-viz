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
                        uint64_t seed,
                        float gamma_fine = 1.0f,
                        float gamma_coarse = 7.0f) {
    
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
            // Coarsest level: initialize with random noise (small σ per details.md:177)
            // Use slightly larger sigma to ensure better initial separation
            random_init_embedding(level.Y, level.num_vertices(), dim, 0.1f, seed);
        } else {
            // Fine level: prolongate from the next coarser level (level_idx+1)
            // level_idx+1 is coarser than level_idx
            prolongate_embedding(hierarchy[level_idx + 1], level, dim);
        }
        
        // Determine epochs for this level
        // details.md line 725: T_l = ceil(500 * |V^l|) per level
        // This implies T_l is the total number of *updates* (steps).
        // Since one "epoch" in optimize_level iterates over all |V^l| groups,
        // the number of epochs corresponds to T_l / |V^l| = 500.
        // Thus, the multiplier (e.g. 500) is the number of epochs (passes).
        uint32_t epochs;
        if (epochs_coarse == 0 && epochs_fine == 0) {
            // Use exact specification from details.md (T=500 independent of N)
            epochs = 500;
        } else {
            // Use provided values as number of epochs directly
            float val = (level_idx == 0) ? static_cast<float>(epochs_fine) : static_cast<float>(epochs_coarse);
            epochs = static_cast<uint32_t>(std::ceil(val));
        }
        
        // Set gamma per paper: γ=1 (finest), γ=7 (coarse)
        // details.md Section 6: "For the finest level: early exaggeration by setting γ=1. For others: γ=7."
        OptimConfig level_cfg = base_cfg;
        if (level_idx == 0) {
            // Finest level
            level_cfg.gamma = gamma_fine;
        } else {
            // Coarse levels
            level_cfg.gamma = gamma_coarse;
        }
        
        // Optimize this level (always with gradient sharing, per paper Section 3.4)
        optimize_level(level, level_cfg, epochs, num_threads, seed + level_idx);
    }
}

} // namespace embed
} // namespace hknn

