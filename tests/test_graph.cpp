#include <catch2/catch_test_macros.hpp>
#include "hknn/graph/csr_graph.hpp"
#include "hknn/embed/gradients.hpp"
#include "hknn/math/kernels.hpp"
#include <vector>
#include <cmath>

TEST_CASE("CSR graph basic operations", "[graph]") {
    hknn::graph::CSR graph;
    
    // Create a simple graph: 3 vertices, triangle
    graph.indptr = {0, 2, 4, 6};
    graph.indices = {1, 2, 0, 2, 0, 1};
    graph.pij = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    
    REQUIRE(graph.num_vertices() == 3);
    REQUIRE(graph.num_edges() == 6);
    REQUIRE(graph.degree(0) == 2);
    REQUIRE(graph.degree(1) == 2);
    REQUIRE(graph.degree(2) == 2);
    
    REQUIRE(graph.has_edge(0, 1));
    REQUIRE(graph.has_edge(1, 0));
    REQUIRE(graph.get_weight(0, 1) == 0.5f);
    
    REQUIRE(graph.validate());
    REQUIRE(graph.is_symmetric());
}

TEST_CASE("CSR graph validation", "[graph]") {
    hknn::graph::CSR graph;
    
    // Invalid: empty graph
    REQUIRE_FALSE(graph.validate());
    
    // Valid: single vertex, no edges
    graph.indptr = {0, 0};
    graph.indices.clear();
    graph.pij.clear();
    REQUIRE(graph.validate());
    
    // Invalid: inconsistent sizes
    graph.indptr = {0, 2};
    graph.indices = {1};
    graph.pij = {0.5f};
    REQUIRE_FALSE(graph.validate());
}

TEST_CASE("Gradient kernels", "[embed]") {
    const int dim = 2;
    float yi[2] = {0.0f, 0.0f};
    float yj[2] = {1.0f, 0.0f};
    float yk[2] = {0.0f, 1.0f};
    
    float gi[2] = {0.0f, 0.0f};
    float pij = 1.0f;
    float gamma = 1.0f;
    
    // Test positive gradient
    // Formula: -2 * p_ij * (y_i - y_j) / (1 + ||y_i - y_j||^2)
    // yi = (0,0), yj = (1,0), so (yi - yj) = (-1, 0)
    // Gradient = -2 * 1.0 * (-1, 0) / (1 + 1) = (2/2, 0) = (1, 0)
    // Since we do y += lr * grad (gradient ascent for maximization),
    // positive gradient moves yi right (towards yj) - correct for attraction
    hknn::embed::grad_pair_pos(gi, yi, yj, pij, dim);
    REQUIRE(gi[0] > 0.0f);  // Positive x direction (towards yj)
    REQUIRE(std::abs(gi[1]) < 1e-5f);
    
    // Reset and test negative gradient
    gi[0] = 0.0f;
    gi[1] = 0.0f;
    // Negative gradient: 2 * gamma * p_ij * (y_i - y_k) / (d^2 * (1 + d^2))
    // yi = (0,0), yk = (0,1), so (yi - yk) = (0, -1), d^2 = 1
    // Gradient = 2 * 1 * 1 * (0, -1) / (1 * 2) = (0, -1)
    // With y += lr * grad, negative gradient moves yi down (away from yk) - correct for repulsion
    hknn::embed::grad_pair_neg(gi, yi, yk, pij, gamma, dim);
    REQUIRE(std::abs(gi[0]) < 1e-5f);
    REQUIRE(gi[1] < 0.0f);  // Negative y direction (away from yk, correct for repulsion)
}

TEST_CASE("Distance computation", "[math]") {
    const int dim = 3;
    float a[3] = {0.0f, 0.0f, 0.0f};
    float b[3] = {1.0f, 2.0f, 3.0f};
    
    float d2 = hknn::squared_distance(a, b, dim);
    float expected = 1.0f + 4.0f + 9.0f;  // 14.0
    REQUIRE(std::abs(d2 - expected) < 1e-6f);
    
    float inv = hknn::inv_one_plus_sqdist(a, b, dim);
    REQUIRE(inv > 0.0f);
    REQUIRE(inv < 1.0f);
}

