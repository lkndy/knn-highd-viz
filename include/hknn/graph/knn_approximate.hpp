#pragma once

#include "csr_graph.hpp"
#include "random_projection_tree.hpp"
#include "nn_descent.hpp"
#include "knn_bruteforce.hpp"  // For symmetrize_knn
#include <vector>

namespace hknn {
namespace graph {

// Build approximate K-NN graph using EFANNA-style approach
// Exactly per details.md pseudocode lines 266-292
CSR build_approx_knn(const float* X, uint32_t N, uint32_t D, uint32_t K,
                     uint32_t num_trees = 4, uint32_t num_passes = 5,
                     uint32_t num_threads = 0, uint64_t seed = 12345) {
    
    // Per details.md line 268: init_neighbors = InitializeByRandomTrees(X, N, D, K)
    std::vector<std::vector<uint32_t>> init_neighbor_indices = 
        initialize_by_random_trees(X, N, D, K, num_trees, 10, seed, num_threads);
    
    // Convert to NeighborList format with distances
    std::vector<NeighborList> neighbors(N);
    for (uint32_t i = 0; i < N; ++i) {
        const float* xi = X + i * D;
        for (uint32_t j : init_neighbor_indices[i]) {
            const float* xj = X + j * D;
            float dist = squared_distance(xi, xj, D);
            neighbors[i].indices.push_back(j);
            neighbors[i].distances.push_back(dist);
        }
        // Sort and keep best K
        std::vector<std::pair<float, uint32_t>> pairs;
        for (size_t idx = 0; idx < neighbors[i].indices.size(); ++idx) {
            pairs.push_back({neighbors[i].distances[idx], neighbors[i].indices[idx]});
        }
        std::sort(pairs.begin(), pairs.end());
        neighbors[i].indices.clear();
        neighbors[i].distances.clear();
        size_t keep = std::min(pairs.size(), static_cast<size_t>(K));
        for (size_t idx = 0; idx < keep; ++idx) {
            neighbors[i].indices.push_back(pairs[idx].second);
            neighbors[i].distances.push_back(pairs[idx].first);
        }
    }
    
    // Per details.md lines 270-283: NN-Descent refinement
    nn_descent_refine(neighbors, X, N, D, K, num_passes, num_threads);
    
    // Per details.md lines 285-291: Build undirected graph
    CSR graph;
    graph.indptr.resize(N + 1, 0);
    
    // Count edges
    for (uint32_t i = 0; i < N; ++i) {
        graph.indptr[i + 1] = graph.indptr[i] + neighbors[i].indices.size();
    }
    
    // Build indices
    graph.indices.reserve(graph.indptr[N]);
    graph.pij.reserve(graph.indptr[N]);
    
    for (uint32_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < neighbors[i].indices.size(); ++j) {
            graph.indices.push_back(neighbors[i].indices[j]);
            graph.pij.push_back(neighbors[i].distances[j]);  // Will be replaced by p_ij computation
        }
    }
    
    // Per details.md: symmetrize to produce undirected graph
    return symmetrize_knn(graph);
}

} // namespace graph
} // namespace hknn

