#include "subspace_loader.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>

bool LoadSubspaceBasisJson(const std::string& path, int max_cols, SubspaceBasisData& out_data, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "Failed to open basis json: " + path;
        return false;
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        error = std::string("Failed to parse json: ") + e.what();
        return false;
    }

    if (!j.contains("basis") || !j["basis"].is_array() || j["basis"].empty()) {
        error = "Missing or empty 'basis' array.";
        return false;
    }

    const auto& basis_j = j["basis"];
    const int rows = static_cast<int>(basis_j.size());
    if (!basis_j[0].is_array() || basis_j[0].empty()) {
        error = "Basis first row is invalid.";
        return false;
    }
    const int cols_in = static_cast<int>(basis_j[0].size());
    const int cols = std::min(std::max(1, max_cols), cols_in);

    out_data = SubspaceBasisData{};
    out_data.rows = rows;
    out_data.cols = cols;
    out_data.basis_row_major.assign(static_cast<size_t>(rows * cols), 0.0f);

    for (int r = 0; r < rows; ++r) {
        const auto& row_j = basis_j[static_cast<size_t>(r)];
        if (!row_j.is_array() || static_cast<int>(row_j.size()) < cols) {
            error = "Basis row length mismatch.";
            return false;
        }
        for (int c = 0; c < cols; ++c) {
            const double v = row_j[static_cast<size_t>(c)].get<double>();
            if (!std::isfinite(v)) {
                error = "Basis contains NaN or Inf.";
                return false;
            }
            out_data.basis_row_major[static_cast<size_t>(r * cols + c)] = static_cast<float>(v);
        }
    }

    if (j.contains("young")) {
        out_data.youngs_modulus = static_cast<float>(j["young"].get<double>());
        out_data.has_material = true;
    }
    if (j.contains("poisson")) {
        out_data.poissons_ratio = static_cast<float>(j["poisson"].get<double>());
        out_data.has_material = true;
    }

    if (j.contains("vertex_mass") && j["vertex_mass"].is_array()) {
        const auto& vm = j["vertex_mass"];
        out_data.vertex_mass.resize(vm.size());
        float min_mass = std::numeric_limits<float>::max();
        float max_mass = 0.0f;
        for (size_t i = 0; i < vm.size(); ++i) {
            const double m = vm[i].get<double>();
            out_data.vertex_mass[i] = static_cast<float>(m);
            min_mass = std::min(min_mass, out_data.vertex_mass[i]);
            max_mass = std::max(max_mass, out_data.vertex_mass[i]);
        }
        out_data.has_vertex_mass = !out_data.vertex_mass.empty() && min_mass > 0.0f && max_mass > 0.0f;
    }

    return true;
}
