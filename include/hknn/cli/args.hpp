#pragma once

#include <string>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <map>
#include <sstream>

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
    // Paper: "T_l = ceil(500 * |V^l|)" per level (details.md line 725)
    // Defaults: 0 means use exact specification, non-zero values are multipliers
    uint32_t epochs_coarse = 0;  // 0 = use T_l = ceil(500 * |V^l|)
    uint32_t epochs_fine = 0;     // 0 = use T_l = ceil(500 * |V^l|)
    uint32_t mneg = 5;
    float gamma = 7.0f;  // Default for coarse (finest uses 1.0, set in hierarchy.hpp)
    float lr = 200.0f;
    std::string schedule = "poly:beta=0.5";
    
    // Output
    int dim = 2;
    uint64_t seed = 123;
    
    // Threading
    uint32_t num_threads = 0;  // 0 = auto-detect
};

// Simple argument parser
Config parse_args(int argc, char* argv[]) {
    Config cfg;
    std::map<std::string, std::string> args;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Hierarchical K-NN Embedding Options:\n";
            std::cout << "  --input <path>        Input data file (.fvecs or raw float32)\n";
            std::cout << "  --N <num>             Number of data points\n";
            std::cout << "  --D <num>             Dimension of data points\n";
            std::cout << "  --out <path>          Output embedding file (base path)\n";
            std::cout << "  --K <num>             Number of nearest neighbors (default: 100)\n";
            std::cout << "  --perplexity <num>    Target perplexity for p_ij (default: 50.0)\n";
            std::cout << "  --kml <num>           Grouping parameter for coarsening (default: 3)\n";
            std::cout << "  --rho <num>           Minimum shrink ratio per level (default: 0.8)\n";
            std::cout << "  --epochs_coarse <num> Multiplier for coarse level epochs (default: 0 = use T_l = ceil(500 * |V^l|))\n";
            std::cout << "  --epochs_fine <num>   Multiplier for finest level epochs (default: 0 = use T_l = ceil(500 * |V^l|))\n";
            std::cout << "  --mneg <num>          Number of negative samples (default: 5)\n";
            std::cout << "  --gamma <num>         Negative sampling balance for coarse levels (default: 7.0, finest uses 1.0)\n";
            std::cout << "  --lr <num>            Learning rate (default: 200.0)\n";
            std::cout << "  --dim <num>           Embedding dimension, 2 or 3 (default: 2)\n";
            std::cout << "  --seed <num>          Random seed (default: 123)\n";
            std::cout << "  --threads <num>       Number of threads, 0 for auto (default: 0)\n";
            std::exit(0);
        }
        
        if (arg.substr(0, 2) == "--") {
            std::string key = arg.substr(2);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args[key] = argv[++i];
            } else {
                args[key] = "1";  // Boolean flag
            }
        }
    }
    
    // Parse values
    auto get_str = [&](const std::string& key, const std::string& def = "") {
        auto it = args.find(key);
        return (it != args.end()) ? it->second : def;
    };
    
    auto get_uint = [&](const std::string& key, uint32_t def = 0) {
        auto it = args.find(key);
        if (it == args.end()) return def;
        return static_cast<uint32_t>(std::stoul(it->second));
    };
    
    auto get_float = [&](const std::string& key, float def = 0.0f) {
        auto it = args.find(key);
        if (it == args.end()) return def;
        return std::stof(it->second);
    };
    
    auto get_int = [&](const std::string& key, int def = 0) {
        auto it = args.find(key);
        if (it == args.end()) return def;
        return std::stoi(it->second);
    };
    
    auto get_uint64 = [&](const std::string& key, uint64_t def = 0) {
        auto it = args.find(key);
        if (it == args.end()) return def;
        return static_cast<uint64_t>(std::stoull(it->second));
    };
    
    cfg.input_path = get_str("input");
    cfg.N = get_uint("N");
    cfg.D = get_uint("D");
    cfg.output_path = get_str("out");
    cfg.K = get_uint("K", 100);
    cfg.perplexity = get_float("perplexity", 50.0f);
    cfg.k_ml = get_uint("kml", 3);
    cfg.rho = get_float("rho", 0.8f);
    cfg.epochs_coarse = get_uint("epochs_coarse", 0);  // 0 = use exact spec from details.md
    cfg.epochs_fine = get_uint("epochs_fine", 0);       // 0 = use exact spec from details.md
    cfg.mneg = get_uint("mneg", 5);
    cfg.gamma = get_float("gamma", 7.0f);  // Default for coarse (finest uses 1.0)
    cfg.lr = get_float("lr", 200.0f);
    cfg.schedule = get_str("schedule", "poly:beta=0.5");
    cfg.dim = get_int("dim", 2);
    cfg.seed = get_uint64("seed", 123);
    cfg.num_threads = get_uint("threads", 0);
    
    // Validate required arguments
    if (cfg.input_path.empty()) {
        throw std::runtime_error("Error: --input is required");
    }
    if (cfg.N == 0) {
        throw std::runtime_error("Error: --N is required");
    }
    if (cfg.D == 0) {
        throw std::runtime_error("Error: --D is required");
    }
    if (cfg.output_path.empty()) {
        throw std::runtime_error("Error: --out is required");
    }
    
    // Validate dimension
    if (cfg.dim != 2 && cfg.dim != 3) {
        throw std::runtime_error("Error: Dimension must be 2 or 3");
    }
    
    return cfg;
}

} // namespace cli
} // namespace hknn
