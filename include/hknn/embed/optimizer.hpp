#pragma once

#include "sharing.hpp"
#include "../graph/coarsen.hpp"
#include "../math/rng.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <numeric>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

namespace hknn {
namespace embed {

// Learning rate schedules
enum class LRSchedule {
    POLYNOMIAL,
    COSINE,
    CONSTANT
};

class LearningRateScheduler {
private:
    LRSchedule schedule_;
    float lr0_;
    float beta_;
    uint32_t T_;
    uint32_t t_;
    
public:
    LearningRateScheduler(LRSchedule sched, float lr0, float beta, uint32_t T)
        : schedule_(sched), lr0_(lr0), beta_(beta), T_(T), t_(0) {}
    
    float get_lr() {
        float lr = lr0_;
        
        switch (schedule_) {
            case LRSchedule::POLYNOMIAL:
                lr = lr0_ / std::pow(1.0f + static_cast<float>(t_) / T_, beta_);
                break;
            case LRSchedule::COSINE:
                lr = lr0_ * 0.5f * (1.0f + std::cos(3.14159265359f * static_cast<float>(t_) / T_));
                break;
            case LRSchedule::CONSTANT:
                lr = lr0_;
                break;
        }
        
        t_++;
        return lr;
    }
    
    void reset() {
        t_ = 0;
    }
};

// SGD optimizer with gradient sharing
void optimize_level(graph::Level& level,
                   const OptimConfig& base_cfg,
                   uint32_t num_epochs,
                   uint32_t num_threads,
                   uint64_t seed) {
    
    if (level.num_vertices() == 0) {
        return;
    }
    
    // Get unique group IDs
    std::vector<uint32_t> groups;
    std::vector<bool> group_seen(level.num_vertices(), false);
    for (uint32_t v = 0; v < level.num_vertices(); ++v) {
        uint32_t gid = level.gid[v];
        if (gid < level.num_vertices() && !group_seen[gid]) {
            groups.push_back(gid);
            group_seen[gid] = true;
        }
    }
    
    if (groups.empty()) {
        // No groups, create one group per vertex
        groups.resize(level.num_vertices());
        std::iota(groups.begin(), groups.end(), 0);
    }
    
    // Learning rate scheduler
    LearningRateScheduler lr_scheduler(LRSchedule::POLYNOMIAL,
                                      base_cfg.lr,
                                      0.5f,  // beta
                                      num_epochs * groups.size());
    
    // Thread pool
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    
    // Per-epoch optimization
    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Shuffle groups for this epoch
        Rng epoch_rng(seed + epoch);
        std::shuffle(groups.begin(), groups.end(), epoch_rng);
        
        // Create thread pool for this epoch
        boost::asio::thread_pool pool(num_threads);
        
        // Create per-thread samplers and RNGs
        std::vector<EdgeSampler> thread_edge_samplers(num_threads);
        std::vector<NegSampler> thread_neg_samplers(num_threads);
        std::vector<Rng> thread_rngs(num_threads);
        
        for (uint32_t t = 0; t < num_threads; ++t) {
            thread_edge_samplers[t].init(level.g);
            thread_neg_samplers[t].init(level.g);
            thread_rngs[t] = RngManager::create_rng_with_offset(seed + epoch, t);
        }
        
        // Process groups in parallel
        const uint32_t chunk_size = std::max(1u, static_cast<uint32_t>(groups.size()) / (num_threads * 4));
        std::mutex config_mutex;
        OptimConfig cfg = base_cfg;
        std::atomic<uint32_t> processed(0);
        
        // Post all tasks
        for (uint32_t chunk_start = 0; chunk_start < groups.size(); chunk_start += chunk_size) {
            uint32_t chunk_end = std::min(chunk_start + chunk_size, static_cast<uint32_t>(groups.size()));
            uint32_t thread_id = (chunk_start / chunk_size) % num_threads;
            
            boost::asio::post(pool, [&, chunk_start, chunk_end, thread_id]() {
                EdgeSampler& local_edge_sampler = thread_edge_samplers[thread_id];
                NegSampler& local_neg_sampler = thread_neg_samplers[thread_id];
                Rng& local_rng = thread_rngs[thread_id];
                
                for (uint32_t g_idx = chunk_start; g_idx < chunk_end; ++g_idx) {
                    uint32_t group_id = groups[g_idx];
                    
                    // Update learning rate
                    OptimConfig local_cfg = cfg;
                    {
                        std::lock_guard<std::mutex> lock(config_mutex);
                        local_cfg.lr = lr_scheduler.get_lr();
                    }
                    
                    // Perform SGD update
                    sgd_group_update(level, level.g, group_id, local_cfg,
                                   local_edge_sampler, local_neg_sampler, local_rng);
                    
                    processed++;
                }
            });
        }
        
        // Wait for all tasks to complete
        pool.join();
    }
}

} // namespace embed
} // namespace hknn

