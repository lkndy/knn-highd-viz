#pragma once

#include "csr_graph.hpp"
#include "../math/kernels.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace hknn {
namespace graph {

// Compute perplexity for a given sigma and distances
float compute_perplexity(const std::vector<float>& distances, float sigma) {
    if (distances.empty()) return 0.0f;
    
    float sum = 0.0f;
    float max_exp = -std::numeric_limits<float>::max();
    
    // Find maximum for numerical stability
    for (float d : distances) {
        float exp_val = -d / (2.0f * sigma * sigma);
        if (exp_val > max_exp) {
            max_exp = exp_val;
        }
    }
    
    // Compute sum of exponentials
    for (float d : distances) {
        float exp_val = -d / (2.0f * sigma * sigma) - max_exp;
        sum += std::exp(exp_val);
    }
    
    if (sum == 0.0f) return 0.0f;
    
    // Compute entropy and perplexity
    float entropy = 0.0f;
    for (float d : distances) {
        float exp_val = -d / (2.0f * sigma * sigma) - max_exp;
        float p = std::exp(exp_val) / sum;
        if (p > 1e-10f) {
            entropy -= p * std::log2(p);
        }
    }
    
    return std::pow(2.0f, entropy);
}

// Binary search for sigma to achieve target perplexity
float find_sigma_binary_search(const std::vector<float>& distances,
                               float target_perplexity,
                               float tolerance = 1e-2f,
                               uint32_t max_iterations = 100) {
    if (distances.empty()) return 1.0f;
    
    float sigma_min = 1e-10f;
    float sigma_max = 1000.0f;
    float sigma = 1.0f;
    
    for (uint32_t iter = 0; iter < max_iterations; ++iter) {
        float perp = compute_perplexity(distances, sigma);
        
        if (std::abs(perp - target_perplexity) < tolerance) {
            break;
        }
        
        if (perp < target_perplexity) {
            sigma_min = sigma;
            sigma = (sigma + sigma_max) / 2.0f;
        } else {
            sigma_max = sigma;
            sigma = (sigma_min + sigma) / 2.0f;
        }
        
        if (sigma_max - sigma_min < 1e-10f) {
            break;
        }
    }
    
    return sigma;
}

// Compute p_{i|j} probabilities for a single vertex
std::vector<float> compute_conditional_probabilities(
    const std::vector<float>& distances,
    float sigma) {
    
    if (distances.empty()) {
        return {};
    }
    
    std::vector<float> probs(distances.size());
    float max_exp = -std::numeric_limits<float>::max();
    
    // Find maximum for numerical stability
    for (float d : distances) {
        float exp_val = -d / (2.0f * sigma * sigma);
        if (exp_val > max_exp) {
            max_exp = exp_val;
        }
    }
    
    // Compute exponentials
    float sum = 0.0f;
    for (size_t i = 0; i < distances.size(); ++i) {
        float exp_val = -distances[i] / (2.0f * sigma * sigma) - max_exp;
        probs[i] = std::exp(exp_val);
        sum += probs[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (float& p : probs) {
            p /= sum;
        }
    }
    
    return probs;
}

// Compute p_{ij} for entire graph (symmetrized)
// Note: Assumes knn_graph is already symmetrized (undirected)
CSR compute_pij(const CSR& knn_graph,
                const float* X,
                uint32_t N,
                uint32_t D,
                float target_perplexity = 50.0f) {
    
    CSR graph = knn_graph;  // Copy structure
    graph.pij.clear();
    graph.pij.resize(graph.indices.size(), 0.0f);
    
    // Step 1: Compute p_{i|j} for each vertex (conditional probabilities)
    std::vector<std::vector<float>> conditional_probs(N);
    std::vector<std::vector<uint32_t>> neighbor_maps(N);  // Map neighbor -> index in conditional_probs
    
    for (uint32_t i = 0; i < N; ++i) {
        auto nbrs = knn_graph.neighbors(i);
        if (nbrs.empty()) continue;
        
        // Get distances to neighbors
        std::vector<float> distances;
        distances.reserve(nbrs.size());
        const float* xi = X + i * D;
        
        for (uint32_t j : nbrs) {
            const float* xj = X + j * D;
            float dist = squared_distance(xi, xj, D);
            distances.push_back(dist);
        }
        
        // Find sigma for target perplexity
        float sigma = find_sigma_binary_search(distances, target_perplexity);
        
        // Compute p_{i|j} (conditional probability of j given i)
        conditional_probs[i] = compute_conditional_probabilities(distances, sigma);
        neighbor_maps[i] = std::vector<uint32_t>(nbrs.begin(), nbrs.end());
    }
    
    // Step 2: Compute symmetric p_{ij} = (p_{i|j} + p_{j|i}) / (2N)
    // For each edge (i, j) in the graph, compute p_ij from both directions
    for (uint32_t i = 0; i < N; ++i) {
        auto nbrs = knn_graph.neighbors(i);
        const auto& p_i_given = conditional_probs[i];
        
        for (size_t j_idx = 0; j_idx < nbrs.size(); ++j_idx) {
            uint32_t j = nbrs[j_idx];
            
            // Get p_{j|i} (probability of j given i)
            float p_j_given_i = p_i_given[j_idx];
            
            // Get p_{i|j} (probability of i given j)
            float p_i_given_j = 0.0f;
            const auto& nbr_map_j = neighbor_maps[j];
            const auto& p_j_given = conditional_probs[j];
            
            // Find index of i in j's neighbor list
            for (size_t k_idx = 0; k_idx < nbr_map_j.size(); ++k_idx) {
                if (nbr_map_j[k_idx] == i) {
                    p_i_given_j = p_j_given[k_idx];
                    break;
                }
            }
            
            // Symmetrize: p_{ij} = (p_{i|j} + p_{j|i}) / (2N)
            float p_ij = (p_i_given_j + p_j_given_i) / (2.0f * N);
            
            // Store in graph at edge (i, j)
            uint32_t edge_idx = graph.indptr[i] + j_idx;
            graph.pij[edge_idx] = p_ij;
        }
    }
    
    // Verify all edges have weights (should be set above)
    for (uint32_t i = 0; i < N; ++i) {
        auto nbrs = graph.neighbors(i);
        for (size_t j_idx = 0; j_idx < nbrs.size(); ++j_idx) {
            uint32_t edge_idx = graph.indptr[i] + j_idx;
            if (graph.pij[edge_idx] <= 0.0f) {
                // Set a small default value to avoid numerical issues
                graph.pij[edge_idx] = 1e-10f;
            }
        }
    }
    
    return graph;
}

} // namespace graph
} // namespace hknn

