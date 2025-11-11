#pragma once

#include "csr_graph.hpp"
#include "../math/kernels.hpp"
#include "../math/rng.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

namespace hknn {
namespace graph {

// Pair of (distance, index) for top-K selection
struct DistanceIndex {
    float distance;
    uint32_t index;
    
    bool operator>(const DistanceIndex& other) const {
        return distance > other.distance;  // For min-heap (we want largest on top)
    }
};

// Top-K selection using min-heap (keep K smallest)
class TopKSelector {
private:
    std::priority_queue<DistanceIndex, std::vector<DistanceIndex>, std::greater<DistanceIndex>> heap_;
    uint32_t K_;
    
public:
    explicit TopKSelector(uint32_t K) : K_(K) {}
    
    void push(float distance, uint32_t index) {
        if (heap_.size() < K_) {
            heap_.push({distance, index});
        } else if (distance < heap_.top().distance) {
            heap_.pop();
            heap_.push({distance, index});
        }
    }
    
    std::vector<uint32_t> get_indices() const {
        std::vector<DistanceIndex> temp;
        auto heap_copy = heap_;
        while (!heap_copy.empty()) {
            temp.push_back(heap_copy.top());
            heap_copy.pop();
        }
        
        // Sort by distance (ascending)
        std::sort(temp.begin(), temp.end(), 
                 [](const DistanceIndex& a, const DistanceIndex& b) {
                     return a.distance < b.distance;
                 });
        
        std::vector<uint32_t> indices;
        indices.reserve(temp.size());
        for (const auto& item : temp) {
            indices.push_back(item.index);
        }
        return indices;
    }
    
    std::vector<float> get_distances() const {
        std::vector<DistanceIndex> temp;
        auto heap_copy = heap_;
        while (!heap_copy.empty()) {
            temp.push_back(heap_copy.top());
            heap_copy.pop();
        }
        
        std::sort(temp.begin(), temp.end(),
                 [](const DistanceIndex& a, const DistanceIndex& b) {
                     return a.distance < b.distance;
                 });
        
        std::vector<float> distances;
        distances.reserve(temp.size());
        for (const auto& item : temp) {
            distances.push_back(item.distance);
        }
        return distances;
    }
};

// Build K-NN graph using brute-force with parallelization
CSR build_knn_bruteforce(const float* X,
                         uint32_t N,
                         uint32_t D,
                         uint32_t K,
                         uint32_t num_threads = 0) {
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    
    CSR graph;
    
    // Initialize CSR structure
    graph.indptr.resize(N + 1, 0);
    std::vector<std::vector<uint32_t>> all_neighbors(N);
    std::vector<std::vector<float>> all_distances(N);
    
    // Thread pool
    boost::asio::thread_pool pool(num_threads);
    std::mutex mutex;
    std::atomic<uint32_t> completed(0);
    
    // Process vertices in parallel
    const uint32_t chunk_size = std::max(1u, N / (num_threads * 4));
    
    for (uint32_t chunk_start = 0; chunk_start < N; chunk_start += chunk_size) {
        uint32_t chunk_end = std::min(chunk_start + chunk_size, N);
        
        boost::asio::post(pool, [&, chunk_start, chunk_end]() {
            for (uint32_t i = chunk_start; i < chunk_end; ++i) {
                TopKSelector selector(K + 1);  // +1 to exclude self
                
                // Compute distances to all other points
                const float* xi = X + i * D;
                for (uint32_t j = 0; j < N; ++j) {
                    if (i == j) continue;  // Skip self
                    
                    const float* xj = X + j * D;
                    float dist = squared_distance(xi, xj, D);
                    selector.push(dist, j);
                }
                
                // Get top-K neighbors
                auto neighbors = selector.get_indices();
                auto distances = selector.get_distances();
                
                // Store results
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    all_neighbors[i] = std::move(neighbors);
                    all_distances[i] = std::move(distances);
                }
                
                completed++;
                if (completed % 1000 == 0) {
                    // Progress indicator (optional)
                }
            }
        });
    }
    
    pool.join();
    
    // Build directed graph (we'll symmetrize later)
    graph.indptr[0] = 0;
    for (uint32_t i = 0; i < N; ++i) {
        graph.indptr[i + 1] = graph.indptr[i] + all_neighbors[i].size();
    }
    
    graph.indices.reserve(graph.indptr[N]);
    for (uint32_t i = 0; i < N; ++i) {
        graph.indices.insert(graph.indices.end(),
                            all_neighbors[i].begin(),
                            all_neighbors[i].end());
    }
    
    // Initialize pij with distances (will be replaced by p_ij computation)
    graph.pij.resize(graph.indices.size(), 0.0f);
    
    return graph;
}

// Symmetrize directed K-NN graph (union of edges)
CSR symmetrize_knn(const CSR& directed_graph) {
    uint32_t N = directed_graph.num_vertices();
    
    CSR sym_graph;
    sym_graph.indptr.resize(N + 1, 0);
    
    // Build edge set (undirected)
    std::vector<std::vector<uint32_t>> neighbors(N);
    std::vector<std::vector<float>> weights(N);
    
    // Add edges from directed graph
    for (uint32_t v = 0; v < N; ++v) {
        auto nbrs = directed_graph.neighbors(v);
        for (uint32_t u : nbrs) {
            // Check if edge already exists
            bool exists = false;
            for (size_t i = 0; i < neighbors[v].size(); ++i) {
                if (neighbors[v][i] == u) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                neighbors[v].push_back(u);
                weights[v].push_back(directed_graph.get_weight(v, u));
                
                // Add reverse edge if not present
                bool reverse_exists = false;
                for (size_t i = 0; i < neighbors[u].size(); ++i) {
                    if (neighbors[u][i] == v) {
                        reverse_exists = true;
                        break;
                    }
                }
                if (!reverse_exists) {
                    neighbors[u].push_back(v);
                    weights[u].push_back(directed_graph.get_weight(v, u));
                }
            }
        }
    }
    
    // Build CSR
    sym_graph.indptr[0] = 0;
    for (uint32_t i = 0; i < N; ++i) {
        sym_graph.indptr[i + 1] = sym_graph.indptr[i] + neighbors[i].size();
    }
    
    sym_graph.indices.reserve(sym_graph.indptr[N]);
    sym_graph.pij.reserve(sym_graph.indptr[N]);
    
    for (uint32_t i = 0; i < N; ++i) {
        sym_graph.indices.insert(sym_graph.indices.end(),
                                neighbors[i].begin(),
                                neighbors[i].end());
        sym_graph.pij.insert(sym_graph.pij.end(),
                            weights[i].begin(),
                            weights[i].end());
    }
    
    return sym_graph;
}

} // namespace graph
} // namespace hknn

