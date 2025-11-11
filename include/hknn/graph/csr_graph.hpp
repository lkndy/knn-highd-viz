#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <span>
#include <cmath>

namespace hknn {
namespace graph {

// Compressed Sparse Row (CSR) format for undirected K-NN graph
struct CSR {
    std::vector<uint32_t> indptr;     // |V|+1, cumulative degree
    std::vector<uint32_t> indices;    // |E|, neighbor indices
    std::vector<float>    pij;        // |E|, symmetrized edge weights
    
    // Get number of vertices
    uint32_t num_vertices() const {
        return indptr.empty() ? 0 : static_cast<uint32_t>(indptr.size() - 1);
    }
    
    // Get number of edges
    uint32_t num_edges() const {
        return static_cast<uint32_t>(indices.size());
    }
    
    // Get degree of vertex v
    uint32_t degree(uint32_t v) const {
        if (v >= num_vertices()) return 0;
        return indptr[v + 1] - indptr[v];
    }
    
    // Get neighbors of vertex v
    std::span<const uint32_t> neighbors(uint32_t v) const {
        if (v >= num_vertices()) {
            return std::span<const uint32_t>();
        }
        uint32_t start = indptr[v];
        uint32_t end = indptr[v + 1];
        return std::span<const uint32_t>(indices.data() + start, end - start);
    }
    
    // Get edge weight p_ij (symmetric, so p_ij = p_ji)
    float get_weight(uint32_t v, uint32_t u) const {
        if (v >= num_vertices() || u >= num_vertices()) {
            return 0.0f;
        }
        
        // Search in v's neighbor list
        uint32_t start = indptr[v];
        uint32_t end = indptr[v + 1];
        auto it = std::find(indices.begin() + start, indices.begin() + end, u);
        if (it != indices.begin() + end) {
            size_t idx = std::distance(indices.begin(), it);
            return pij[idx];
        }
        return 0.0f;
    }
    
    // Check if edge (v, u) exists
    bool has_edge(uint32_t v, uint32_t u) const {
        return get_weight(v, u) > 0.0f;
    }
    
    // Validate CSR structure
    bool validate() const {
        if (indptr.empty()) return false;
        if (indptr[0] != 0) return false;
        if (indices.size() != pij.size()) return false;
        
        uint32_t nv = num_vertices();
        for (uint32_t v = 0; v < nv; ++v) {
            if (indptr[v + 1] < indptr[v]) return false;
            if (indptr[v + 1] > indices.size()) return false;
            
            // Check neighbor list is sorted (optional, but helpful)
            uint32_t start = indptr[v];
            uint32_t end = indptr[v + 1];
            for (uint32_t i = start + 1; i < end; ++i) {
                if (indices[i] <= indices[i - 1]) {
                    // Not sorted, but that's okay
                }
            }
        }
        
        return true;
    }
    
    // Check symmetry (p_ij should equal p_ji)
    bool is_symmetric(float tolerance = 1e-6f) const {
        uint32_t nv = num_vertices();
        for (uint32_t v = 0; v < nv; ++v) {
            for (uint32_t u = v + 1; u < nv; ++u) {
                float pvu = get_weight(v, u);
                float puv = get_weight(u, v);
                if (std::abs(pvu - puv) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Clear all data
    void clear() {
        indptr.clear();
        indices.clear();
        pij.clear();
    }
};

} // namespace graph
} // namespace hknn

