#include "hknn/cli/args.hpp"
#include "hknn/io/reader.hpp"
#include "hknn/io/writer.hpp"
#include "hknn/graph/knn_bruteforce.hpp"
#include "hknn/graph/pij.hpp"
#include "hknn/graph/coarsen.hpp"
#include "hknn/embed/hierarchy.hpp"
#include <iostream>
#include <chrono>
#include <boost/timer/timer.hpp>

using namespace hknn;

int main(int argc, char* argv[]) {
    try {
        // Parse command-line arguments
        cli::Config cfg = cli::parse_args(argc, argv);
        
        std::cout << "=== Hierarchical K-NN Embedding ===\n";
        std::cout << "Input: " << cfg.input_path << "\n";
        std::cout << "N: " << cfg.N << ", D: " << cfg.D << "\n";
        std::cout << "Output: " << cfg.output_path << "\n";
        std::cout << "K: " << cfg.K << ", perplexity: " << cfg.perplexity << "\n";
        std::cout << "Dimension: " << cfg.dim << ", Seed: " << cfg.seed << "\n\n";
        
        boost::timer::cpu_timer timer;
        
        // Load data
        std::cout << "Loading data...\n";
        timer.start();
        auto loader = io::create_loader(cfg.input_path);
        if (!loader->load(cfg.input_path, cfg.N, cfg.D)) {
            std::cerr << "Error: Failed to load data from " << cfg.input_path << "\n";
            return 1;
        }
        timer.stop();
        std::cout << "Data loaded in " << timer.format() << "\n\n";
        
        const float* X = loader->data();
        
        // Build K-NN graph
        std::cout << "Building K-NN graph (brute-force)...\n";
        timer.start();
        graph::CSR knn_graph = graph::build_knn_bruteforce(X, cfg.N, cfg.D, cfg.K, cfg.num_threads);
        timer.stop();
        std::cout << "K-NN graph built in " << timer.format() << "\n";
        std::cout << "Vertices: " << knn_graph.num_vertices() << ", Edges: " << knn_graph.num_edges() << "\n\n";
        
        // Symmetrize graph
        std::cout << "Symmetrizing graph...\n";
        timer.start();
        knn_graph = graph::symmetrize_knn(knn_graph);
        timer.stop();
        std::cout << "Graph symmetrized in " << timer.format() << "\n";
        std::cout << "Edges after symmetrization: " << knn_graph.num_edges() << "\n\n";
        
        // Compute p_ij
        std::cout << "Computing p_ij probabilities...\n";
        timer.start();
        knn_graph = graph::compute_pij(knn_graph, X, cfg.N, cfg.D, cfg.perplexity);
        timer.stop();
        std::cout << "p_ij computed in " << timer.format() << "\n\n";
        
        // Build hierarchy
        std::cout << "Building hierarchy...\n";
        timer.start();
        std::vector<graph::Level> hierarchy = graph::build_hierarchy(
            knn_graph, cfg.k_ml, cfg.rho, cfg.seed);
        timer.stop();
        std::cout << "Hierarchy built in " << timer.format() << "\n";
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
        timer.start();
        embed::OptimConfig optim_cfg;
        optim_cfg.lr = cfg.lr;
        optim_cfg.gamma = cfg.gamma;
        optim_cfg.M = cfg.mneg;
        optim_cfg.dim = cfg.dim;
        optim_cfg.max_grad_norm = 10.0f;
        
        embed::refine_hierarchical(hierarchy, optim_cfg,
                                  cfg.epochs_coarse, cfg.epochs_fine,
                                  cfg.num_threads, cfg.seed);
        timer.stop();
        std::cout << "Refinement completed in " << timer.format() << "\n\n";
        
        // Get finest level embedding
        if (hierarchy.empty()) {
            std::cerr << "Error: No levels in hierarchy\n";
            return 1;
        }
        
        const graph::Level& finest = hierarchy[0];
        const float* Y = finest.Y.data();
        
        // Write output
        std::cout << "Writing output...\n";
        timer.start();
        if (!io::write_embedding(cfg.output_path, Y, cfg.N, cfg.dim)) {
            std::cerr << "Error: Failed to write output\n";
            return 1;
        }
        timer.stop();
        std::cout << "Output written in " << timer.format() << "\n\n";
        
        std::cout << "=== Done ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

