#pragma once

#include <string>
#include <cstdint>
#include <boost/program_options.hpp>

namespace hknn {
namespace cli {

struct Config {
    // I/O
    std::string input_path;
    uint32_t N = 0;
    uint32_t D = 0;
    std::string output_path;
    
    // K-NN construction
    uint32_t K = 100;
    float perplexity = 50.0f;
    
    // Coarsening
    bool levels_auto = true;
    uint32_t k_ml = 3;
    float rho = 0.8f;
    
    // Optimization
    uint32_t epochs_coarse = 4;
    uint32_t epochs_fine = 8;
    uint32_t mneg = 5;
    float gamma = 5.0f;
    float lr = 200.0f;
    std::string schedule = "poly:beta=0.5";
    
    // Output
    int dim = 2;
    uint64_t seed = 123;
    
    // Threading
    uint32_t num_threads = 0;  // 0 = auto-detect
};

Config parse_args(int argc, char* argv[]) {
    namespace po = boost::program_options;
    
    Config cfg;
    po::options_description desc("Hierarchical K-NN Embedding Options");
    
    desc.add_options()
        ("help,h", "Print help message")
        ("input", po::value<std::string>(&cfg.input_path)->required(), "Input data file (.fvecs or raw float32)")
        ("N", po::value<uint32_t>(&cfg.N)->required(), "Number of data points")
        ("D", po::value<uint32_t>(&cfg.D)->required(), "Dimension of data points")
        ("out", po::value<std::string>(&cfg.output_path)->required(), "Output embedding file (base path)")
        
        ("K", po::value<uint32_t>(&cfg.K)->default_value(100), "Number of nearest neighbors")
        ("perplexity", po::value<float>(&cfg.perplexity)->default_value(50.0f), "Target perplexity for p_ij")
        
        ("levels_auto", po::value<bool>(&cfg.levels_auto)->default_value(true), "Auto-build hierarchy levels")
        ("kml", po::value<uint32_t>(&cfg.k_ml)->default_value(3), "Grouping parameter for coarsening")
        ("rho", po::value<float>(&cfg.rho)->default_value(0.8f), "Minimum shrink ratio per level")
        
        ("epochs_coarse", po::value<uint32_t>(&cfg.epochs_coarse)->default_value(4), "Epochs per coarse level")
        ("epochs_fine", po::value<uint32_t>(&cfg.epochs_fine)->default_value(8), "Epochs for finest level")
        ("mneg", po::value<uint32_t>(&cfg.mneg)->default_value(5), "Number of negative samples")
        ("gamma", po::value<float>(&cfg.gamma)->default_value(5.0f), "Negative sampling balance")
        ("lr", po::value<float>(&cfg.lr)->default_value(200.0f), "Learning rate")
        ("schedule", po::value<std::string>(&cfg.schedule)->default_value("poly:beta=0.5"), "Learning rate schedule")
        
        ("dim", po::value<int>(&cfg.dim)->default_value(2), "Embedding dimension (2 or 3)")
        ("seed", po::value<uint64_t>(&cfg.seed)->default_value(123), "Random seed")
        
        ("threads", po::value<uint32_t>(&cfg.num_threads)->default_value(0), "Number of threads (0 = auto)");
    
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        
        if (vm.count("help")) {
            std::cout << desc << "\n";
            std::exit(0);
        }
        
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        std::exit(1);
    }
    
    // Validate dimension
    if (cfg.dim != 2 && cfg.dim != 3) {
        throw std::runtime_error("Dimension must be 2 or 3");
    }
    
    return cfg;
}

} // namespace cli
} // namespace hknn

