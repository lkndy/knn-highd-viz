#include <catch2/catch_test_macros.hpp>
#include "hknn/embed/sampler.hpp"
#include "hknn/graph/csr_graph.hpp"
#include <vector>
#include <cmath>

TEST_CASE("Alias table", "[sampler]") {
    hknn::embed::AliasTable alias;
    
    // Test uniform distribution
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};
    alias.build(weights);
    REQUIRE(alias.size() == 4);
    
    // Test sampling (should work without crashing)
    hknn::Rng rng(123);
    for (int i = 0; i < 100; ++i) {
        uint32_t sample = alias.draw(rng);
        REQUIRE(sample < 4);
    }
}

TEST_CASE("Negative sampler", "[sampler]") {
    hknn::graph::CSR graph;
    graph.indptr = {0, 2, 4, 5};
    graph.indices = {1, 2, 0, 2, 0};
    graph.pij = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    
    hknn::embed::NegSampler sampler;
    sampler.init(graph);
    
    hknn::Rng rng(123);
    for (int i = 0; i < 100; ++i) {
        uint32_t sample = sampler.sample(rng);
        REQUIRE(sample < 3);
    }
}

TEST_CASE("Edge sampler", "[sampler]") {
    hknn::graph::CSR graph;
    graph.indptr = {0, 2, 4, 5};
    graph.indices = {1, 2, 0, 2, 0};
    graph.pij = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    
    hknn::embed::EdgeSampler sampler;
    sampler.init(graph);
    
    hknn::Rng rng(123);
    for (int i = 0; i < 100; ++i) {
        auto edge = sampler.sample_edge_from_vertex(0, rng);
        REQUIRE(edge.first == 0);
        REQUIRE(edge.second < 3);
    }
}

