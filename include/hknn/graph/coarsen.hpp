#pragma once

#include "csr_graph.hpp"
#include "../math/rng.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <climits>
#include <boost/dynamic_bitset.hpp>
#include <random>

namespace hknn {
namespace graph {

// Level structure for hierarchical representation
struct Level {
    CSR g;                             // graph at level l
    std::vector<uint32_t> gid;         // |V|, group id for each vertex
    std::vector<uint32_t> parent;      // |V_l| -> vertex in V_{l+1} (coarser level)
    std::vector<std::vector<uint32_t>> children; // inverse mapping (children of each vertex in coarser level)
    std::vector<float> Y;              // |V_l| * dim, embedding coordinates (SoA format)
    
    uint32_t num_vertices() const {
        return g.num_vertices();
    }
    
    void clear() {
        g.clear();
        gid.clear();
        parent.clear();
        children.clear();
        Y.clear();
    }
};

// Coarsen graph by one level
// Returns coarser level and updates parent/children mappings
Level coarsen_level(const Level& fine_level,
                   uint32_t k_ml,  // grouping parameter (number of neighbors to pull)
                   float rho,      // minimum shrink ratio
                   Rng& rng) {
    
    const CSR& graph = fine_level.g;
    uint32_t N = graph.num_vertices();
    
    if (N == 0) {
        return Level{};
    }
    
    Level coarse_level;
    
    // Step 1: Shuffle vertices
    std::vector<uint32_t> vertex_order(N);
    std::iota(vertex_order.begin(), vertex_order.end(), 0);
    std::shuffle(vertex_order.begin(), vertex_order.end(), rng);
    
    // Step 2: Greedy grouping
    boost::dynamic_bitset<> assigned(N, false);
    std::vector<uint32_t> groups;  // Group representatives (vertices in coarse level)
    coarse_level.parent.resize(N, UINT32_MAX);
    coarse_level.gid.resize(N, UINT32_MAX);
    
    uint32_t next_group_id = 0;
    
    for (uint32_t seed : vertex_order) {
        if (assigned[seed]) continue;
        
        // Start a new group with seed
        uint32_t group_id = next_group_id++;
        groups.push_back(seed);
        assigned[seed] = true;
        coarse_level.parent[seed] = group_id;
        coarse_level.gid[seed] = group_id;
        
        // Pull k_ml neighbors that are still free
        auto nbrs = graph.neighbors(seed);
        uint32_t pulled = 0;
        
        for (uint32_t nbr : nbrs) {
            if (pulled >= k_ml) break;
            if (!assigned[nbr]) {
                assigned[nbr] = true;
                coarse_level.parent[nbr] = group_id;
                coarse_level.gid[nbr] = group_id;
                pulled++;
            }
        }
    }
    
    uint32_t N_coarse = groups.size();
    
    // Check shrink ratio
    if (N_coarse >= static_cast<uint32_t>(rho * N)) {
        // Not enough shrinkage, return empty level
        return Level{};
    }
    
    // Step 3: Build coarse graph
    // Contract groups: add edge (g_u, g_v) if any edge exists across member sets
    coarse_level.g.indptr.resize(N_coarse + 1, 0);
    
    // Build adjacency for coarse graph
    std::vector<std::vector<uint32_t>> coarse_adj(N_coarse);
    std::vector<std::vector<float>> coarse_weights(N_coarse);
    boost::dynamic_bitset<> coarse_edge_exists(N_coarse * N_coarse, false);
    
    // Map fine vertices to coarse vertices
    std::vector<uint32_t> fine_to_coarse(N, UINT32_MAX);
    for (uint32_t i = 0; i < N; ++i) {
        if (coarse_level.parent[i] != UINT32_MAX) {
            fine_to_coarse[i] = coarse_level.parent[i];
        }
    }
    
    // Scan all edges in fine graph
    for (uint32_t v = 0; v < N; ++v) {
        uint32_t g_v = fine_to_coarse[v];
        if (g_v == UINT32_MAX) continue;
        
        auto nbrs = graph.neighbors(v);
        for (uint32_t u : nbrs) {
            uint32_t g_u = fine_to_coarse[u];
            if (g_u == UINT32_MAX) continue;
            if (g_u == g_v) continue;  // Skip self-loops in coarse graph
            
            // Check if edge already exists
            uint32_t edge_key = g_v * N_coarse + g_u;
            if (!coarse_edge_exists[edge_key]) {
                coarse_edge_exists[edge_key] = true;
                coarse_adj[g_v].push_back(g_u);
                // Use average weight (or max, or sum - paper doesn't specify, use max for now)
                float w = graph.get_weight(v, u);
                coarse_weights[g_v].push_back(w);
            } else {
                // Update weight (take max)
                for (size_t i = 0; i < coarse_adj[g_v].size(); ++i) {
                    if (coarse_adj[g_v][i] == g_u) {
                        coarse_weights[g_v][i] = std::max(coarse_weights[g_v][i], graph.get_weight(v, u));
                        break;
                    }
                }
            }
        }
    }
    
    // Build CSR for coarse graph
    coarse_level.g.indptr[0] = 0;
    for (uint32_t i = 0; i < N_coarse; ++i) {
        coarse_level.g.indptr[i + 1] = coarse_level.g.indptr[i] + coarse_adj[i].size();
    }
    
    coarse_level.g.indices.reserve(coarse_level.g.indptr[N_coarse]);
    coarse_level.g.pij.reserve(coarse_level.g.indptr[N_coarse]);
    
    for (uint32_t i = 0; i < N_coarse; ++i) {
        coarse_level.g.indices.insert(coarse_level.g.indices.end(),
                                     coarse_adj[i].begin(),
                                     coarse_adj[i].end());
        coarse_level.g.pij.insert(coarse_level.g.pij.end(),
                                 coarse_weights[i].begin(),
                                 coarse_weights[i].end());
    }
    
    // Step 4: Build children mapping (for prolongation)
    coarse_level.children.resize(N_coarse);
    for (uint32_t v = 0; v < N; ++v) {
        if (coarse_level.parent[v] != UINT32_MAX) {
            coarse_level.children[coarse_level.parent[v]].push_back(v);
        }
    }
    
    return coarse_level;
}

// Build hierarchy of levels
std::vector<Level> build_hierarchy(const CSR& graph,
                                  uint32_t k_ml,
                                  float rho,
                                  uint64_t seed) {
    
    std::vector<Level> hierarchy;
    
    // Initialize finest level (level 0)
    Level fine_level;
    fine_level.g = graph;
    fine_level.gid.resize(graph.num_vertices());
    std::iota(fine_level.gid.begin(), fine_level.gid.end(), 0);
    fine_level.parent.resize(graph.num_vertices(), UINT32_MAX);
    fine_level.children.clear();  // No children at finest level
    
    Rng rng(seed);
    Level current_level = fine_level;
    hierarchy.push_back(current_level);
    
    // Coarsen until stopping condition
    uint32_t level_idx = 0;
    while (true) {
        Level coarse_level = coarsen_level(current_level, k_ml, rho, rng);
        
        if (coarse_level.num_vertices() == 0 || 
            coarse_level.num_vertices() >= current_level.num_vertices()) {
            // No more coarsening possible
            break;
        }
        
        // The coarse_level.parent maps fine vertices to coarse vertices
        // Store this parent mapping in the fine level (current_level)
        current_level.parent = coarse_level.parent;
        
        // Set up children mapping in coarse level (for potential future use)
        coarse_level.children.resize(coarse_level.num_vertices());
        for (uint32_t v = 0; v < current_level.num_vertices(); ++v) {
            if (coarse_level.parent[v] != UINT32_MAX) {
                coarse_level.children[coarse_level.parent[v]].push_back(v);
            }
        }
        
        // Clear parent mapping in coarse level (it's only for fine->coarse mapping)
        coarse_level.parent.clear();
        coarse_level.parent.resize(coarse_level.num_vertices(), UINT32_MAX);
        
        // Update hierarchy: update fine level with parent mapping, add coarse level
        hierarchy[level_idx] = current_level;
        hierarchy.push_back(coarse_level);
        
        level_idx++;
        current_level = coarse_level;
        
        // Stop if graph is small enough
        if (coarse_level.num_vertices() < 10) {
            break;
        }
        
        // Limit hierarchy depth
        if (level_idx >= 10) {
            break;
        }
    }
    
    return hierarchy;
}

} // namespace graph
} // namespace hknn

