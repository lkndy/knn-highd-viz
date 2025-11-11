#pragma once

#include "../graph/csr_graph.hpp"
#include "../math/rng.hpp"
#include <vector>
#include <numeric>

namespace hknn {
namespace embed {

// Alias table for O(1) sampling from a probability distribution
class AliasTable {
private:
    std::vector<float> prob_;
    std::vector<uint32_t> alias_;
    uint32_t n_;
    
public:
    AliasTable() : n_(0) {}
    
    // Build alias table from weights
    void build(const std::vector<double>& weights) {
        n_ = static_cast<uint32_t>(weights.size());
        if (n_ == 0) {
            prob_.clear();
            alias_.clear();
            return;
        }
        
        prob_.resize(n_);
        alias_.resize(n_);
        
        // Normalize weights
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum == 0.0) {
            // Uniform distribution
            for (uint32_t i = 0; i < n_; ++i) {
                prob_[i] = 1.0f / n_;
                alias_[i] = i;
            }
            return;
        }
        
        std::vector<double> normalized(n_);
        for (uint32_t i = 0; i < n_; ++i) {
            normalized[i] = weights[i] / sum * n_;
        }
        
        // Build alias table
        std::vector<uint32_t> small, large;
        for (uint32_t i = 0; i < n_; ++i) {
            if (normalized[i] < 1.0) {
                small.push_back(i);
            } else {
                large.push_back(i);
            }
        }
        
        while (!small.empty() && !large.empty()) {
            uint32_t less = small.back();
            small.pop_back();
            uint32_t more = large.back();
            large.pop_back();
            
            prob_[less] = static_cast<float>(normalized[less]);
            alias_[less] = more;
            
            normalized[more] -= (1.0 - normalized[less]);
            if (normalized[more] < 1.0) {
                small.push_back(more);
            } else {
                large.push_back(more);
            }
        }
        
        while (!large.empty()) {
            uint32_t i = large.back();
            large.pop_back();
            prob_[i] = 1.0f;
            alias_[i] = i;
        }
        
        while (!small.empty()) {
            uint32_t i = small.back();
            small.pop_back();
            prob_[i] = 1.0f;
            alias_[i] = i;
        }
    }
    
    // Sample from alias table
    uint32_t draw(Rng& rng) const {
        if (n_ == 0) return 0;
        
        uint32_t i = RngManager::uniform_int(rng, 0u, n_ - 1);
        float u = RngManager::uniform_real(rng);
        
        if (u < prob_[i]) {
            return i;
        } else {
            return alias_[i];
        }
    }
    
    uint32_t size() const {
        return n_;
    }
};

// Edge sampler for positive samples (proportional to p_ij)
// Note: This is a per-vertex sampler that samples edges from a specific vertex's neighbors
class EdgeSampler {
private:
    const graph::CSR* csr_;
    
public:
    EdgeSampler() : csr_(nullptr) {}
    
    void init(const graph::CSR& graph) {
        csr_ = &graph;
    }
    
    // Sample an edge from vertex v's neighbors (proportional to p_ij)
    std::pair<uint32_t, uint32_t> sample_edge_from_vertex(uint32_t v, Rng& rng) const {
        if (!csr_ || v >= csr_->num_vertices()) {
            return {v, v};
        }
        
        auto nbrs = csr_->neighbors(v);
        if (nbrs.empty()) {
            return {v, v};
        }
        
        // Build alias table for this vertex's neighbors
        std::vector<double> weights;
        for (uint32_t u : nbrs) {
            float w = csr_->get_weight(v, u);
            weights.push_back(static_cast<double>(w));
        }
        
        AliasTable sampler;
        sampler.build(weights);
        uint32_t idx = sampler.draw(rng);
        
        return {v, nbrs[idx]};
    }
};

// Negative sampler (degree-proportional)
class NegSampler {
private:
    AliasTable alias_;
    uint32_t num_vertices_;
    
public:
    NegSampler() : num_vertices_(0) {}
    
    void init(const graph::CSR& graph) {
        num_vertices_ = graph.num_vertices();
        std::vector<double> weights(num_vertices_);
        
        for (uint32_t i = 0; i < num_vertices_; ++i) {
            weights[i] = static_cast<double>(graph.degree(i));
        }
        
        alias_.build(weights);
    }
    
    uint32_t sample(Rng& rng) const {
        if (num_vertices_ == 0) return 0;
        return alias_.draw(rng);
    }
};

} // namespace embed
} // namespace hknn

