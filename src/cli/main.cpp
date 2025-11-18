#include "hknn/cli/args.hpp"
#include "hknn/io/reader.hpp"
#include "hknn/io/writer.hpp"
#include "hknn/graph/knn_approximate.hpp"
#include "hknn/graph/pij.hpp"
#include "hknn/graph/coarsen.hpp"
#include "hknn/embed/hierarchy.hpp"
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
        std::cout << "Timer created.\n" << std::flush;
        
        // Load data
        std::cout << "Loading data...\n" << std::flush;
        auto load_start = std::chrono::high_resolution_clock::now();
        std::cout << "Creating loader...\n" << std::flush;
        auto loader = io::create_loader(cfg.input_path);
        std::cout << "Loader created.\n" << std::flush;
        if (!loader->load(cfg.input_path, cfg.N, cfg.D)) {
            std::cerr << "Error: Failed to load data from " << cfg.input_path << "\n";
            return 1;
        }
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
        std::cout << "Data loaded in " << load_duration.count() << " ms\n\n";
        
        const float* X = loader->data();
        
        // Build K-NN graph (approximate EFANNA-style per details.md Section 5)
        std::cout << "Building approximate K-NN graph (EFANNA-style)...\n";
        auto knn_start = std::chrono::high_resolution_clock::now();
        graph::CSR knn_graph = graph::build_approx_knn(X, cfg.N, cfg.D, cfg.K, 
                                                        4,  // num_trees (typical EFANNA: 4-8)
                                                        5,  // num_passes (typical NN-Descent: 3-5)
                                                        cfg.num_threads,
                                                        cfg.seed);
        auto knn_end = std::chrono::high_resolution_clock::now();
        auto knn_duration = std::chrono::duration_cast<std::chrono::milliseconds>(knn_end - knn_start);
        std::cout << "K-NN graph built in " << knn_duration.count() << " ms\n";
        std::cout << "Vertices: " << knn_graph.num_vertices() << ", Edges: " << knn_graph.num_edges() << "\n";
        std::cout << "(Graph is already symmetrized per details.md)\n\n";
        
        // Compute p_ij
        std::cout << "Computing p_ij probabilities...\n";
        auto pij_start = std::chrono::high_resolution_clock::now();
        knn_graph = graph::compute_pij(knn_graph, X, cfg.N, cfg.D, cfg.perplexity);
        auto pij_end = std::chrono::high_resolution_clock::now();
        auto pij_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pij_end - pij_start);
        std::cout << "p_ij computed in " << pij_duration.count() << " ms\n\n";
        
        // Build hierarchy
        std::cout << "Building hierarchy...\n";
        auto hier_start = std::chrono::high_resolution_clock::now();
        std::vector<graph::Level> hierarchy = graph::build_hierarchy(
            knn_graph, cfg.k_ml, cfg.rho, cfg.seed);
        auto hier_end = std::chrono::high_resolution_clock::now();
        auto hier_duration = std::chrono::duration_cast<std::chrono::milliseconds>(hier_end - hier_start);
        std::cout << "Hierarchy built in " << hier_duration.count() << " ms\n";
        std::cout << "Number of levels: " << hierarchy.size() << "\n";
        for (size_t i = 0; i < hierarchy.size(); ++i) {
            std::cout << "  Level " << i << ": " << hierarchy[i].num_vertices() << " vertices\n";
        }
        std::cout << "\n";
        
        // Set up group IDs for all levels
        // At each level, groups are determined by the coarsening process
        // For the finest level, we use single-vertex groups for gradient sharing
        if (!hierarchy.empty()) {
            // Finest level: each vertex is its own group (for gradient sharing within neighborhoods)
            hierarchy[0].gid.resize(hierarchy[0].num_vertices());
            for (uint32_t v = 0; v < hierarchy[0].num_vertices(); ++v) {
                hierarchy[0].gid[v] = v;
            }
            
            // Coarser levels: groups are the vertices themselves (already set by coarsening)
            for (size_t i = 1; i < hierarchy.size(); ++i) {
                if (hierarchy[i].gid.size() != hierarchy[i].num_vertices()) {
                    hierarchy[i].gid.resize(hierarchy[i].num_vertices());
                    for (uint32_t v = 0; v < hierarchy[i].num_vertices(); ++v) {
                        hierarchy[i].gid[v] = v;
                    }
                }
            }
        }
        
        // Hierarchical refinement
        std::cout << "Starting hierarchical refinement...\n";
        auto refine_start = std::chrono::high_resolution_clock::now();
        embed::OptimConfig optim_cfg;
        optim_cfg.lr = cfg.lr;
        optim_cfg.gamma = cfg.gamma;
        optim_cfg.M = cfg.mneg;
        optim_cfg.dim = cfg.dim;
        optim_cfg.max_grad_norm = 10.0f;
        
        embed::refine_hierarchical(hierarchy, optim_cfg,
                                  cfg.epochs_coarse, cfg.epochs_fine,
                                  cfg.num_threads, cfg.seed);
        auto refine_end = std::chrono::high_resolution_clock::now();
        auto refine_duration = std::chrono::duration_cast<std::chrono::milliseconds>(refine_end - refine_start);
        std::cout << "Refinement completed in " << refine_duration.count() << " ms\n\n";
        
        // Get finest level embedding
        if (hierarchy.empty()) {
            std::cerr << "Error: No levels in hierarchy\n";
            return 1;
        }
        
        const graph::Level& finest = hierarchy[0];
        const float* Y = finest.Y.data();
        
        // Write output
        std::cout << "Writing output...\n";
        auto write_start = std::chrono::high_resolution_clock::now();
        if (!io::write_embedding(cfg.output_path, Y, cfg.N, cfg.dim)) {
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

