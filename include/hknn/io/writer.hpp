#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>

namespace hknn {
namespace io {

// Write raw float32 row-major format
bool write_f32(const std::string& path, 
               const float* data, 
               uint32_t N, 
               uint32_t dim) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(data), N * dim * sizeof(float));
    return file.good();
}

// Write CSV format (id,x,y[,z])
bool write_csv(const std::string& path,
               const float* data,
               uint32_t N,
               uint32_t dim) {
    std::ofstream file(path);
    if (!file) {
        return false;
    }
    
    // Write header
    if (dim == 2) {
        file << "id,x,y\n";
    } else if (dim == 3) {
        file << "id,x,y,z\n";
    } else {
        return false;  // Only support 2D and 3D
    }
    
    // Write data
    file << std::fixed << std::setprecision(6);
    for (uint32_t i = 0; i < N; ++i) {
        file << i;
        for (uint32_t d = 0; d < dim; ++d) {
            file << "," << data[i * dim + d];
        }
        file << "\n";
    }
    
    return file.good();
}

// Write both formats
bool write_embedding(const std::string& base_path,
                     const float* data,
                     uint32_t N,
                     uint32_t dim) {
    // Write .f32 file
    std::string f32_path = base_path;
    if (f32_path.find('.') == std::string::npos) {
        f32_path += ".f32";
    }
    if (!write_f32(f32_path, data, N, dim)) {
        return false;
    }
    
    // Write .csv file
    std::string csv_path = base_path;
    size_t dot_pos = csv_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        csv_path = csv_path.substr(0, dot_pos);
    }
    csv_path += ".csv";
    
    return write_csv(csv_path, data, N, dim);
}

} // namespace io
} // namespace hknn

