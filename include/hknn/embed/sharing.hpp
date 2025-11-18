#pragma once

#include "../graph/coarsen.hpp"
#include "sampler.hpp"
#include "gradients.hpp"
#include "../math/rng.hpp"
#include <vector>
#include <algorithm>

namespace hknn {
namespace embed {

// Configuration for optimization
struct OptimConfig {
    float lr = 200.0f;           // Learning rate
    float gamma = 5.0f;          // Negative sampling balance
    uint32_t M = 5;              // Number of negative samples
    float max_grad_norm = 10.0f; // Gradient clipping
    int dim = 2;                 // Embedding dimension (2 or 3)
};

// Pick representative for a group (smallest index or max degree)
uint32_t pick_representative(const graph::Level& level, uint32_t group_id) {
    // Find all vertices in this group
    std::vector<uint32_t> members;
    for (uint32_t v = 0; v < level.num_vertices(); ++v) {
        if (level.gid[v] == group_id) {
            members.push_back(v);
        }
    }
    
    if (members.empty()) {
        return 0;
    }
    
    // Pick the one with maximum degree (more connections = better representative)
    uint32_t best = members[0];
    uint32_t best_degree = level.g.degree(best);
    
    for (uint32_t v : members) {
        uint32_t deg = level.g.degree(v);
        if (deg > best_degree) {
            best_degree = deg;
            best = v;
        }
    }
    
    return best;
}

// Get all members of a group
std::vector<uint32_t> get_group_members(const graph::Level& level, uint32_t group_id) {
    std::vector<uint32_t> members;
    for (uint32_t v = 0; v < level.num_vertices(); ++v) {
        if (level.gid[v] == group_id) {
            members.push_back(v);
        }
    }
    return members;
}

// Sample positive edge for a vertex (from its neighbors)
// This is now handled by EdgeSampler::sample_edge_from_vertex

// SGD update for a single group with gradient sharing
void sgd_group_update(graph::Level& level,
                     const graph::CSR& /* graph */,
                     uint32_t group_id,
                     const OptimConfig& cfg,
                     EdgeSampler& edge_sampler,
                     NegSampler& neg_sampler,
                     Rng& rng) {
    
    // Get group members
    std::vector<uint32_t> members = get_group_members(level, group_id);
    if (members.empty()) {
        return;
    }
    
    // Pick representative
    uint32_t rep = pick_representative(level, group_id);
    
    // Sample positive edge for representative
    auto edge = edge_sampler.sample_edge_from_vertex(rep, rng);
    uint32_t i = edge.first;
    uint32_t j = edge.second;
    float pij = level.g.get_weight(i, j);
    if (pij <= 0.0f) {
        pij = 1e-10f;  // Small default
    }
    
    // Sample negative vertices
    std::vector<uint32_t> negatives(cfg.M);
    for (uint32_t k = 0; k < cfg.M; ++k) {
        negatives[k] = neg_sampler.sample(rng);
        // Avoid sampling the positive neighbor
        while (negatives[k] == j && cfg.M > 1) {
            negatives[k] = neg_sampler.sample(rng);
        }
    }
    
    // Compute gradient for representative
    float gi[3] = {0.0f, 0.0f, 0.0f};
    const float* yi = level.Y.data() + i * cfg.dim;
    const float* yj = level.Y.data() + j * cfg.dim;
    
    // Positive gradient
    grad_pair_pos(gi, yi, yj, pij, cfg.dim);
    
    // Negative gradients
    for (uint32_t k = 0; k < cfg.M; ++k) {
        const float* yk = level.Y.data() + negatives[k] * cfg.dim;
        grad_pair_neg(gi, yi, yk, pij, cfg.gamma, cfg.dim);
    }
    
    // Clip gradient
    clip_gradient(gi, cfg.dim, cfg.max_grad_norm);
    
    // Fan-out: apply gradient to all group members
    // details.md Section 11.2 line 681: y[u] = y[u] - lr * gy
    // Apply gradient exactly as specified (no variation)
    for (uint32_t u : members) {
        for (int t = 0; t < cfg.dim; ++t) {
            level.Y[u * cfg.dim + t] -= cfg.lr * gi[t];
        }
    }
}

} // namespace embed
} // namespace hknn

