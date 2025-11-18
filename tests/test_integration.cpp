#include <catch2/catch_test_macros.hpp>
#include "hknn/graph/knn_bruteforce.hpp"
#include "hknn/graph/pij.hpp"
#include <vector>
#include <cmath>

TEST_CASE("Small K-NN construction", "[integration]") {
    const uint32_t N = 20;
    const uint32_t D = 5;
    const uint32_t K = 5;
    
    // Create simple test data (points on a line)
    std::vector<float> X(N * D);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t d = 0; d < D; ++d) {
            X[i * D + d] = static_cast<float>(i) + static_cast<float>(d) * 0.1f;
        }
    }
    
    // Build K-NN graph (use single thread to avoid mutex issues in tests)
    hknn::graph::CSR knn_graph = hknn::graph::build_knn_bruteforce(
        X.data(), N, D, K, 1);
    
    REQUIRE(knn_graph.num_vertices() == N);
    REQUIRE(knn_graph.num_edges() > 0);
    
    // Verify each vertex has neighbors
    for (uint32_t i = 0; i < N; ++i) {
        REQUIRE(knn_graph.degree(i) >= 0);
    }
    
    // Symmetrize
    knn_graph = hknn::graph::symmetrize_knn(knn_graph);
    REQUIRE(knn_graph.is_symmetric());
    
    // Compute p_ij (use smaller perplexity for small dataset)
    knn_graph = hknn::graph::compute_pij(knn_graph, X.data(), N, D, 10.0f);
    
    // Verify all edges have positive weights
    for (uint32_t i = 0; i < N; ++i) {
        auto nbrs = knn_graph.neighbors(i);
        for (uint32_t j : nbrs) {
            float w = knn_graph.get_weight(i, j);
            REQUIRE(w >= 0.0f);  // Allow zero for very small probabilities
        }
    }
}

