#include "hknn/cli/args.hpp"
#include "hknn/io/reader.hpp"
#include "hknn/io/writer.hpp"
#include "hknn/api.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace hknn;

int main(int argc, char* argv[]) {
    try {
        std::cout << "Starting HKNN embedding...\n" << std::flush;
        
        // Parse command-line arguments
        std::cout << "Parsing arguments...\n" << std::flush;
        cli::Config cfg = cli::parse_args(argc, argv);
        std::cout << "Arguments parsed.\n" << std::flush;
        
        std::cout << "=== Hierarchical K-NN Embedding ===\n" << std::flush;
        std::cout << "Input: " << cfg.input_path << "\n" << std::flush;
        std::cout << "N: " << cfg.N << ", D: " << cfg.D << "\n" << std::flush;
        std::cout << "Output: " << cfg.output_path << "\n" << std::flush;
        std::cout << "K: " << cfg.K << ", perplexity: " << cfg.perplexity << "\n" << std::flush;
        std::cout << "Dimension: " << cfg.dim << ", Seed: " << cfg.seed << "\n\n" << std::flush;
        
        std::cout << "Creating timer...\n" << std::flush;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Load data
        std::cout << "Loading data...\n" << std::flush;
        auto load_start = std::chrono::high_resolution_clock::now();
        auto loader = io::create_loader(cfg.input_path);
        if (!loader->load(cfg.input_path, cfg.N, cfg.D)) {
            std::cerr << "Error: Failed to load data from " << cfg.input_path << "\n";
            return 1;
        }
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
        std::cout << "Data loaded in " << load_duration.count() << " ms\n\n";
        
        const float* X = loader->data();
        
        // Run Hierarchical Embedding
        std::cout << "Running hierarchical embedding...\n";
        auto embed_start = std::chrono::high_resolution_clock::now();
        
        // Pass 1.0f for gamma_fine as per paper/defaults, and cfg.gamma (default 7.0) for gamma_coarse
        Embedding embedding = run_hierarchy_embedding(
            X, cfg.N, cfg.D,
            cfg.K,
            cfg.dim,
            cfg.perplexity,
            0, // max_levels auto
            cfg.mneg,
            cfg.gamma, // gamma_coarse
            1.0f,      // gamma_fine
            cfg.lr,
            cfg.seed,
            cfg.num_threads
        );
        
        auto embed_end = std::chrono::high_resolution_clock::now();
        auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(embed_end - embed_start);
        std::cout << "Embedding completed in " << embed_duration.count() << " ms\n\n";
        
        // Write output
        std::cout << "Writing output...\n";
        auto write_start = std::chrono::high_resolution_clock::now();
        if (!io::write_embedding(cfg.output_path, embedding.data.data(), embedding.N, embedding.dim)) {
            std::cerr << "Error: Failed to write output\n";
            return 1;
        }
        auto write_end = std::chrono::high_resolution_clock::now();
        auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);
        std::cout << "Output written in " << write_duration.count() << " ms\n\n";
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - start_time);
        std::cout << "Total time: " << total_duration.count() << " ms\n";
        
        std::cout << "=== Done ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
