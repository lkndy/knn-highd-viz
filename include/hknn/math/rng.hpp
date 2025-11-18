#pragma once

#include <cstdint>
#include <random>

namespace hknn {

using Rng = std::mt19937_64;

// RNG utilities for reproducibility
class RngManager {
public:
    static Rng create_rng(uint64_t seed) {
        Rng rng(seed);
        return rng;
    }
    
    static Rng create_rng_with_offset(uint64_t base_seed, uint32_t thread_id) {
        Rng rng(base_seed);
        // Jump ahead for thread-specific RNG
        for (uint32_t i = 0; i < thread_id; ++i) {
            rng.discard(1000000);  // Simple jump ahead
        }
        return rng;
    }
    
    template<typename T>
    static T uniform_int(Rng& rng, T min, T max) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng);
    }
    
    static float uniform_real(Rng& rng, float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng);
    }
    
    static float normal(Rng& rng, float mean = 0.0f, float stddev = 1.0f) {
        std::normal_distribution<float> dist(mean, stddev);
        return dist(rng);
    }
};

} // namespace hknn

