#pragma once

#include "../math/rng.hpp"
#include "../util/thread_pool.hpp"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>

namespace hknn {
namespace graph {

// Random projection tree node
struct RPTreeNode {
    std::vector<uint32_t> indices;  // Point indices in this node
    std::vector<float> projection;  // Random projection vector
    float threshold;                 // Split threshold
    RPTreeNode* left;
    RPTreeNode* right;
    bool is_leaf;
    
    RPTreeNode() : threshold(0.0f), left(nullptr), right(nullptr), is_leaf(false) {}
    
    ~RPTreeNode() {
        delete left;
        delete right;
    }
};

// Random projection tree for approximate K-NN initialization
class RandomProjectionTree {
private:
    const float* X_;
    uint32_t N_;
    uint32_t D_;
    RPTreeNode* root_;
    uint64_t seed_;
    uint32_t max_leaf_size_;
    
    // Generate sparse random projection vector (1/3 non-zero, ±1)
    std::vector<float> generate_projection(Rng& rng) {
        std::vector<float> proj(D_);
        uint32_t num_nonzero = D_ / 3;
        if (num_nonzero == 0) num_nonzero = 1;
        
        // Initialize to zero
        std::fill(proj.begin(), proj.end(), 0.0f);
        
        // Set random positions to ±1
        std::unordered_set<uint32_t> used;
        for (uint32_t i = 0; i < num_nonzero; ++i) {
            uint32_t pos;
            do {
                pos = RngManager::uniform_int(rng, 0u, D_ - 1);
            } while (used.find(pos) != used.end());
            used.insert(pos);
            
            proj[pos] = (RngManager::uniform_real(rng) < 0.5f) ? -1.0f : 1.0f;
        }
        
        return proj;
    }
    
    // Project point onto vector
    float project_point(const float* x, const std::vector<float>& proj) {
        float result = 0.0f;
        for (uint32_t d = 0; d < D_; ++d) {
            result += x[d] * proj[d];
        }
        return result;
    }
    
    // Build tree recursively
    RPTreeNode* build_tree_recursive(const std::vector<uint32_t>& indices, 
                                     uint32_t depth,
                                     Rng& rng) {
        if (indices.size() <= max_leaf_size_ || depth > 20) {
            RPTreeNode* node = new RPTreeNode();
            node->indices = indices;
            node->is_leaf = true;
            return node;
        }
        
        RPTreeNode* node = new RPTreeNode();
        node->projection = generate_projection(rng);
        
        // Compute projections and find median
        std::vector<float> projections;
        projections.reserve(indices.size());
        for (uint32_t idx : indices) {
            const float* x = X_ + idx * D_;
            projections.push_back(project_point(x, node->projection));
        }
        
        // Find median
        std::vector<float> sorted_proj = projections;
        std::sort(sorted_proj.begin(), sorted_proj.end());
        node->threshold = sorted_proj[sorted_proj.size() / 2];
        
        // Split points
        std::vector<uint32_t> left_indices, right_indices;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (projections[i] < node->threshold) {
                left_indices.push_back(indices[i]);
            } else {
                right_indices.push_back(indices[i]);
            }
        }
        
        // Ensure both children have points
        if (left_indices.empty()) {
            left_indices.push_back(right_indices.back());
            right_indices.pop_back();
        }
        if (right_indices.empty()) {
            right_indices.push_back(left_indices.back());
            left_indices.pop_back();
        }
        
        node->left = build_tree_recursive(left_indices, depth + 1, rng);
        node->right = build_tree_recursive(right_indices, depth + 1, rng);
        
        return node;
    }
    
    // Collect points from leaf containing point i
    void collect_from_leaf(RPTreeNode* node, uint32_t point_idx, 
                          std::unordered_set<uint32_t>& candidates) {
        if (!node) return;
        
        if (node->is_leaf) {
            for (uint32_t idx : node->indices) {
                if (idx != point_idx) {
                    candidates.insert(idx);
                }
            }
            return;
        }
        
        // Traverse to appropriate leaf
        const float* x = X_ + point_idx * D_;
        float proj_val = project_point(x, node->projection);
        
        if (proj_val < node->threshold) {
            collect_from_leaf(node->left, point_idx, candidates);
        } else {
            collect_from_leaf(node->right, point_idx, candidates);
        }
    }
    
public:
    RandomProjectionTree(const float* X, uint32_t N, uint32_t D, 
                        uint32_t max_leaf_size = 10, uint64_t seed = 12345)
        : X_(X), N_(N), D_(D), root_(nullptr), seed_(seed), max_leaf_size_(max_leaf_size) {
        std::vector<uint32_t> all_indices(N);
        for (uint32_t i = 0; i < N; ++i) {
            all_indices[i] = i;
        }
        
        Rng rng(seed);
        root_ = build_tree_recursive(all_indices, 0, rng);
    }
    
    ~RandomProjectionTree() {
        delete root_;
    }
    
    // Collect candidate neighbors for point i from this tree
    void collect_candidates(uint32_t i, std::unordered_set<uint32_t>& candidates) {
        collect_from_leaf(root_, i, candidates);
    }
};

// Build multiple random projection trees and collect initial candidates
// Returns: for each point i, a set of candidate neighbor indices
std::vector<std::vector<uint32_t>> initialize_by_random_trees(
    const float* X, uint32_t N, uint32_t D, uint32_t K,
    uint32_t num_trees = 4, uint32_t max_leaf_size = 10, uint64_t seed = 12345,
    uint32_t num_threads = 0) {
    
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    
    std::vector<std::vector<uint32_t>> init_neighbors(N);
    std::mutex mutex;
    
    // Build trees and collect candidates (parallelize tree building)
    util::ThreadPool pool(num_threads);
    
    for (uint32_t t = 0; t < num_trees; ++t) {
        pool.post([&, t]() {
            RandomProjectionTree tree(X, N, D, max_leaf_size, seed + t);
            
            // Collect candidates for all points from this tree
            std::vector<std::vector<uint32_t>> tree_candidates(N);
            for (uint32_t i = 0; i < N; ++i) {
                std::unordered_set<uint32_t> candidates;
                tree.collect_candidates(i, candidates);
                tree_candidates[i].assign(candidates.begin(), candidates.end());
            }
            
            // Merge into global init_neighbors
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (uint32_t i = 0; i < N; ++i) {
                    for (uint32_t idx : tree_candidates[i]) {
                        if (std::find(init_neighbors[i].begin(), init_neighbors[i].end(), idx) == 
                            init_neighbors[i].end()) {
                            init_neighbors[i].push_back(idx);
                        }
                    }
                }
            }
        });
    }
    
    pool.join();
    
    // If we don't have enough candidates, add random ones
    Rng rng(seed + num_trees);
    for (uint32_t i = 0; i < N; ++i) {
        while (init_neighbors[i].size() < static_cast<size_t>(K)) {
            uint32_t random_idx = RngManager::uniform_int(rng, 0u, N - 1);
            if (random_idx != i && 
                std::find(init_neighbors[i].begin(), init_neighbors[i].end(), random_idx) == 
                init_neighbors[i].end()) {
                init_neighbors[i].push_back(random_idx);
            }
        }
        
        // Limit to reasonable size (2*K for NN-Descent)
        if (init_neighbors[i].size() > static_cast<size_t>(2 * K)) {
            init_neighbors[i].resize(2 * K);
        }
    }
    
    return init_neighbors;
}

} // namespace graph
} // namespace hknn

