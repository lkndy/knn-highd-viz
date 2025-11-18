#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <span>
#include <filesystem>

namespace hknn {
namespace io {

// Data loader interface
class DataLoader {
public:
    virtual ~DataLoader() = default;
    virtual bool load(const std::string& path, uint32_t N, uint32_t D) = 0;
    virtual std::span<const float> get_row(uint32_t idx) const = 0;
    virtual const float* data() const = 0;
    virtual uint32_t num_rows() const = 0;
    virtual uint32_t num_dims() const = 0;
};

// Raw float32 row-major loader
class RawFloat32Loader : public DataLoader {
private:
    std::vector<float> data_;
    uint32_t N_;
    uint32_t D_;
    uint32_t stride_;  // Optional stride (defaults to D)
    uint32_t offset_;  // Optional header offset
    
public:
    RawFloat32Loader(uint32_t stride = 0, uint32_t offset = 0)
        : stride_(stride), offset_(offset) {}
    
    bool load(const std::string& path, uint32_t N, uint32_t D) override {
        N_ = N;
        D_ = D;
        if (stride_ == 0) stride_ = D;
        
        std::filesystem::path file_path(path);
        if (!std::filesystem::exists(file_path)) {
            return false;
        }
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }
        
        // Seek to offset
        file.seekg(offset_ * sizeof(float));
        
        // Read data
        size_t total_floats = N * stride_;
        data_.resize(total_floats);
        file.read(reinterpret_cast<char*>(data_.data()), 
                  total_floats * sizeof(float));
        
        return file.good();
    }
    
    std::span<const float> get_row(uint32_t idx) const override {
        if (idx >= N_) {
            return std::span<const float>();
        }
        return std::span<const float>(data_.data() + idx * stride_, D_);
    }
    
    const float* data() const override {
        return data_.data();
    }
    
    uint32_t num_rows() const override {
        return N_;
    }
    
    uint32_t num_dims() const override {
        return D_;
    }
};

// .fvecs format loader (FAISS style)
class FvecsLoader : public DataLoader {
private:
    std::vector<float> data_;
    uint32_t N_;
    uint32_t D_;
    
public:
    bool load(const std::string& path, uint32_t N, uint32_t D) override {
        N_ = N;
        D_ = D;
        
        std::filesystem::path file_path(path);
        if (!std::filesystem::exists(file_path)) {
            return false;
        }
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }
        
        data_.reserve(N * D);
        
        for (uint32_t i = 0; i < N; ++i) {
            // Read dimension header
            int32_t d;
            file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
            
            if (file.eof()) {
                return false;
            }
            
            // Validate dimension
            if (static_cast<uint32_t>(d) != D) {
                return false;
            }
            
            // Read vector data
            std::vector<float> row(D);
            file.read(reinterpret_cast<char*>(row.data()), D * sizeof(float));
            
            if (file.eof()) {
                return false;
            }
            
            data_.insert(data_.end(), row.begin(), row.end());
        }
        
        return true;
    }
    
    std::span<const float> get_row(uint32_t idx) const override {
        if (idx >= N_) {
            return std::span<const float>();
        }
        return std::span<const float>(data_.data() + idx * D_, D_);
    }
    
    const float* data() const override {
        return data_.data();
    }
    
    uint32_t num_rows() const override {
        return N_;
    }
    
    uint32_t num_dims() const override {
        return D_;
    }
};

// Factory function to create appropriate loader
std::unique_ptr<DataLoader> create_loader(const std::string& path) {
    std::filesystem::path file_path(path);
    std::string ext = file_path.extension().string();
    
    if (ext == ".fvecs") {
        return std::make_unique<FvecsLoader>();
    } else {
        // Default to raw float32
        return std::make_unique<RawFloat32Loader>();
    }
}

} // namespace io
} // namespace hknn
