#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

namespace hknn {
namespace util {

// Simple thread pool using std::thread
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    std::atomic<bool> stop_{false};
    size_t active_workers_ = 0;
    
public:
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                        ++active_workers_;
                    }
                    task();
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        --active_workers_;
                        if (tasks_.empty() && active_workers_ == 0) {
                            finished_.notify_all();
                        }
                    }
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    template<typename F>
    void post(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                return;
            }
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }
    
    void join() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        finished_.wait(lock, [this] { return tasks_.empty() && active_workers_ == 0; });
    }
};

} // namespace util
} // namespace hknn

