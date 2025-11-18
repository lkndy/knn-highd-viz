#include <catch2/catch_test_macros.hpp>
#include "hknn/io/reader.hpp"
#include "hknn/io/writer.hpp"
#include <fstream>
#include <vector>
#include <cmath>

TEST_CASE("Raw float32 loader", "[io]") {
    // Create test data
    const uint32_t N = 100;
    const uint32_t D = 10;
    std::vector<float> data(N * D);
    for (uint32_t i = 0; i < N * D; ++i) {
        data[i] = static_cast<float>(i) * 0.1f;
    }
    
    // Write test file
    std::string test_file = "test_data.raw";
    std::ofstream file(test_file, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), N * D * sizeof(float));
    file.close();
    
    // Load test file
    hknn::io::RawFloat32Loader loader;
    REQUIRE(loader.load(test_file, N, D));
    REQUIRE(loader.num_rows() == N);
    REQUIRE(loader.num_dims() == D);
    
    // Verify data
    for (uint32_t i = 0; i < N; ++i) {
        auto row = loader.get_row(i);
        REQUIRE(row.size() == D);
        for (uint32_t d = 0; d < D; ++d) {
            REQUIRE(std::abs(row[d] - data[i * D + d]) < 1e-6f);
        }
    }
    
    // Cleanup
    std::remove(test_file.c_str());
}

TEST_CASE("CSV writer", "[io]") {
    const uint32_t N = 10;
    const uint32_t dim = 2;
    std::vector<float> embedding(N * dim);
    for (uint32_t i = 0; i < N; ++i) {
        embedding[i * dim + 0] = static_cast<float>(i);
        embedding[i * dim + 1] = static_cast<float>(i) * 2.0f;
    }
    
    std::string test_file = "test_output.csv";
    REQUIRE(hknn::io::write_csv(test_file, embedding.data(), N, dim));
    
    // Verify file exists and has correct content
    std::ifstream file(test_file);
    REQUIRE(file.good());
    
    std::string line;
    std::getline(file, line);
    REQUIRE(line == "id,x,y");
    
    uint32_t count = 0;
    while (std::getline(file, line) && count < N) {
        count++;
    }
    REQUIRE(count == N);
    
    // Cleanup
    file.close();
    std::remove(test_file.c_str());
}

