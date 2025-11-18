#pragma once

#include "random_projection_tree.hpp"
#include "../math/kernels.hpp"
#include "../util/thread_pool.hpp"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <mutex>
#include <limits>
#include <thread>

namespace hknn {
namespace graph {

// Neighbor list with distances
struct NeighborList {
    std::vector<uint32_t> indices;
    std::vector<float> distances;
    
    // Get worst (largest) distance
    float worst_distance() const {
        if (distances.empty()) return std::numeric_limits<float>::max();
        return *std::max_element(distances.begin(), distances.end());
    }
    
    // Insert neighbor if it improves the list (maintains K best)
    void insert_if_better(uint32_t idx, float dist, uint32_t K) {
        // Check if already present
        auto it = std::find(indices.begin(), indices.end(), idx);
        if (it != indices.end()) {
            size_t pos = std::distance(indices.begin(), it);
            if (dist < distances[pos]) {
                distances[pos] = dist;
                // Re-sort to maintain order
                std::vector<std::pair<float, uint32_t>> pairs;
                for (size_t i = 0; i < indices.size(); ++i) {
                    pairs.push_back({distances[i], indices[i]});
                }
                std::sort(pairs.begin(), pairs.end());
                for (size_t i = 0; i < indices.size(); ++i) {
                    indices[i] = pairs[i].second;
                    distances[i] = pairs[i].first;
                }
            }
            return;
        }
        
        // Add if list not full or distance is better than worst
        if (indices.size() < static_cast<size_t>(K) || dist < worst_distance()) {
            indices.push_back(idx);
            distances.push_back(dist);
            
            // Sort by distance and keep only K best
            std::vector<std::pair<float, uint32_t>> pairs;
            for (size_t i = 0; i < indices.size(); ++i) {
                pairs.push_back({distances[i], indices[i]});
            }
            std::sort(pairs.begin(), pairs.end());
            
            indices.clear();
            distances.clear();
            size_t keep = std::min(pairs.size(), static_cast<size_t>(K));
            for (size_t i = 0; i < keep; ++i) {
                indices.push_back(pairs[i].second);
                distances.push_back(pairs[i].first);
            }
        }
    }
};

// NN-Descent refinement per details.md pseudocode lines 271-283
// neighbors: input/output neighbor lists (per point)
void nn_descent_refine(std::vector<NeighborList>& neighbors,
                      const float* X, uint32_t N, uint32_t D, uint32_t K,
                      uint32_t num_passes = 5, uint32_t num_threads = 0) {
    
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    
    // Per details.md line 271: for pass = 1 to P
    for (uint32_t pass = 0; pass < num_passes; ++pass) {
        // Per details.md line 272: neighbors_new = neighbors
        std::vector<NeighborList> neighbors_new = neighbors;
        
        // Per details.md line 273: for i in 0..N-1 (parallel)
        util::ThreadPool pool(num_threads);
        std::vector<std::mutex> vertex_mutexes(N);  // Per-vertex mutexes to reduce contention
        
        const uint32_t chunk_size = std::max(1u, N / (num_threads * 4));
        
        for (uint32_t chunk_start = 0; chunk_start < N; chunk_start += chunk_size) {
            uint32_t chunk_end = std::min(chunk_start + chunk_size, N);
            
            pool.post([&, chunk_start, chunk_end]() {
                for (uint32_t i = chunk_start; i < chunk_end; ++i) {
                    // Build hash set for fast lookup (per details.md line 277: if k not in neighbors_new[i])
                    std::unordered_set<uint32_t> current_neighbors(neighbors_new[i].indices.begin(), 
                                                                   neighbors_new[i].indices.end());
                    
                    // Per details.md line 274: for each j in neighbors[i]
                    for (size_t j_idx = 0; j_idx < neighbors[i].indices.size(); ++j_idx) {
                        uint32_t j = neighbors[i].indices[j_idx];
                        
                        // Per details.md line 275: for each k in neighbors[j]
                        for (size_t k_idx = 0; k_idx < neighbors[j].indices.size(); ++k_idx) {
                            uint32_t k = neighbors[j].indices[k_idx];
                            
                            // Per details.md line 276: if j == k: continue
                            if (j == k) continue;
                            
                            // Per details.md line 277: if k not in neighbors_new[i]
                            if (current_neighbors.find(k) == current_neighbors.end()) {
                                // Per details.md line 278: dist = ||x_i - x_k||^2
                                const float* xi = X + i * D;
                                const float* xk = X + k * D;
                                float dist = squared_distance(xi, xk, D);
                                
                                // Per details.md line 279-282: insert if improves
                                {
                                    std::lock_guard<std::mutex> lock(vertex_mutexes[i]);
                                    if (current_neighbors.find(k) == current_neighbors.end()) {
                                        neighbors_new[i].insert_if_better(k, dist, K);
                                        current_neighbors.insert(k);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        
        pool.join();
        
        // Per details.md line 283: neighbors = neighbors_new
        neighbors = std::move(neighbors_new);
    }
}

} // namespace graph
} // namespace hknn

