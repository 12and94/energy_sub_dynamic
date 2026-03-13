#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "sim_params.h"
#include "subspace_simulator.h"

namespace {

bool EndsWith(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

bool StartsWith(const std::string& value, const std::string& prefix) {
    if (prefix.size() > value.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), value.begin());
}

std::string FindMeshInDir(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> msh_candidates;
    std::vector<std::filesystem::path> json_mesh_candidates;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto p = entry.path();
        const auto ext = p.extension().string();
        const auto filename = p.filename().string();
        if (ext == ".msh") {
            msh_candidates.push_back(p);
        } else if (ext == ".json" && !EndsWith(filename, "_result.json")) {
            json_mesh_candidates.push_back(p);
        }
    }
    auto pick_first = [](std::vector<std::filesystem::path>& v) -> std::string {
        if (v.empty()) return {};
        std::sort(v.begin(), v.end());
        return v.front().string();
    };
    const std::string msh = pick_first(msh_candidates);
    if (!msh.empty()) return msh;
    return pick_first(json_mesh_candidates);
}

std::string FindBasisInDir(const std::filesystem::path& dir, const std::string& mesh_stem) {
    std::vector<std::filesystem::path> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto p = entry.path();
        if (p.extension() != ".json") {
            continue;
        }
        const std::string filename = p.filename().string();
        if (!EndsWith(filename, "_result.json")) {
            continue;
        }
        if (!mesh_stem.empty() && StartsWith(filename, mesh_stem)) {
            return p.string();
        }
        candidates.push_back(p);
    }
    if (candidates.empty()) {
        return {};
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front().string();
}

} // namespace

int main(int argc, char** argv) {
    std::string mesh_path = "../Cpp_Cuda/data/example_1/ball.msh";
    std::string basis_path = "../Cpp_Cuda/data/example_1/ball_ck_lam57692.307692_mu38461.538462_alpha1.000000_rho1000.000000_result.json";
    int frames = 200;
    bool no_obj = false;
    bool download_only = false;
    bool render = false;
    std::vector<std::string> positional_args;
    positional_args.reserve(static_cast<std::size_t>(std::max(0, argc - 1)));

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-obj") {
            no_obj = true;
            continue;
        }
        if (arg == "--download-only") {
            download_only = true;
            continue;
        }
        if (arg == "--render") {
            render = true;
            continue;
        }
        positional_args.push_back(arg);
    }

    if (no_obj && download_only) {
        std::cerr << "Options --no-obj and --download-only are mutually exclusive.\n";
        return 1;
    }

    if (!positional_args.empty()) {
        mesh_path = positional_args[0];
    }
    if (positional_args.size() > 1) {
        basis_path = positional_args[1];
    }
    if (positional_args.size() > 2) {
        frames = std::max(1, std::atoi(positional_args[2].c_str()));
    }

    std::filesystem::path input_path(mesh_path);
    if (std::filesystem::is_directory(input_path)) {
        const std::string resolved_mesh = FindMeshInDir(input_path);
        if (resolved_mesh.empty()) {
            std::cerr << "No mesh found in directory: " << mesh_path << "\n";
            return 1;
        }
        mesh_path = resolved_mesh;
        if (basis_path.empty()) {
            basis_path = FindBasisInDir(input_path, std::filesystem::path(mesh_path).stem().string());
        }
    }

    if (basis_path.empty()) {
        const auto parent = std::filesystem::path(mesh_path).parent_path();
        basis_path = FindBasisInDir(parent, std::filesystem::path(mesh_path).stem().string());
    }
    if (basis_path.empty()) {
        std::cerr << "Failed to resolve basis json path.\n";
        return 1;
    }

    SimParams params;
    ComputeDerivedParams(params);

    SubspaceSimulator simulator;
    std::string error;
    if (!simulator.Initialize(mesh_path, basis_path, params, error)) {
        std::cerr << "Initialize failed: " << error << "\n";
        return 1;
    }
    std::cout << "Vulkan compute device: " << simulator.GetGpuDeviceName() << "\n";

    const bool write_obj_frames = !no_obj && !download_only;
    const bool render_frames = render;
    const bool download_frames = download_only || write_obj_frames || render_frames;
    const std::string output_dir = mesh_path + ".vkcs_output";
    const auto sim_begin = std::chrono::steady_clock::now();
    if (!simulator.Run(frames, output_dir, download_frames, write_obj_frames, render_frames, error)) {
        std::cerr << "Run failed: " << error << "\n";
        return 1;
    }
    const auto sim_end = std::chrono::steady_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sim_end - sim_begin).count();
    const double total_s = static_cast<double>(total_ms) / 1000.0;
    const auto& stats = simulator.GetTimingStats();

    const double known_ms = stats.predict_ms +
                            stats.upload_xstar_ms +
                            stats.reconstruct_newton_ms +
                            stats.gradient_ms +
                            stats.linear_solve_ms +
                            stats.reconstruct_final_ms +
                            stats.download_x_ms +
                            stats.render_ms +
                            stats.update_velocity_ms +
                            stats.write_obj_ms;
    const double other_ms = std::max(0.0, static_cast<double>(total_ms) - known_ms);
    const double solve_overhead_ms = std::max(0.0, stats.linear_solve_ms - stats.hessian_ms);

    std::cout << "Total simulation time: " << total_ms << " ms (" << total_s << " s)\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Timing breakdown (ms):\n";
    std::cout << "  predict: " << stats.predict_ms << "\n";
    std::cout << "  upload_xstar: " << stats.upload_xstar_ms << "\n";
    std::cout << "  reconstruct_newton: " << stats.reconstruct_newton_ms << "\n";
    std::cout << "  gradient_reduced: " << stats.gradient_ms << "\n";
    std::cout << "  linear_solve_total: " << stats.linear_solve_ms << "\n";
    std::cout << "  hessian_reduced_calls: " << stats.hessian_ms << "\n";
    std::cout << "  linear_solve_overhead: " << solve_overhead_ms << "\n";
    std::cout << "  reconstruct_final: " << stats.reconstruct_final_ms << "\n";
    std::cout << "  download_x: " << stats.download_x_ms << "\n";
    std::cout << "  render: " << stats.render_ms << "\n";
    std::cout << "  update_velocity: " << stats.update_velocity_ms << "\n";
    std::cout << "  write_obj: " << stats.write_obj_ms << "\n";
    std::cout << "  other: " << other_ms << "\n";
    std::cout << "Iterations/counts:\n";
    std::cout << "  substeps: " << stats.substeps << "\n";
    std::cout << "  newton_iterations: " << stats.newton_iterations << "\n";
    std::cout << "  cg_iterations: " << stats.cg_iterations << "\n";
    std::cout << "  hessian_calls: " << stats.hessian_calls << "\n";
    if (no_obj) {
        std::cout << "Done. Output skipped (--no-obj)\n";
    } else if (download_only) {
        std::cout << "Done. Download only (--download-only), no OBJ output.\n";
    } else {
        std::cout << "Done. Output: " << output_dir << "\n";
    }
    if (render_frames) {
        std::cout << "Render mode: realtime window (--render)\n";
        std::cout << "Camera controls: RMB drag=orbit, mouse wheel=zoom, R=reset, Esc=quit render loop\n";
    }
    return 0;
}
