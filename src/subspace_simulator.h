#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sim_params.h"
#include "types.h"
#include "vk_two_phase.h"

class SubspaceSimulator {
public:
    struct TimingStats {
        double predict_ms = 0.0;
        double upload_xstar_ms = 0.0;
        double reconstruct_newton_ms = 0.0;
        double gradient_ms = 0.0;
        double linear_solve_ms = 0.0;
        double hessian_ms = 0.0;
        double reconstruct_final_ms = 0.0;
        double download_x_ms = 0.0;
        double render_ms = 0.0;
        double update_velocity_ms = 0.0;
        double write_obj_ms = 0.0;
        std::uint64_t substeps = 0;
        std::uint64_t newton_iterations = 0;
        std::uint64_t cg_iterations = 0;
        std::uint64_t hessian_calls = 0;
    };

    bool Initialize(const std::string& mesh_path,
                    const std::string& basis_path,
                    SimParams params,
                    std::string& error);

    bool Run(int total_frames,
             const std::string& output_dir,
             bool download_frames,
             bool write_obj_frames,
             bool render_frames,
             std::string& error);
    const TimingStats& GetTimingStats() const { return timing_stats_; }
    const std::string& GetGpuDeviceName() const { return gpu_.DeviceName(); }

private:
    bool Substep(std::string& error);

    bool BuildSurfaceTriangles();
    bool WriteObj(const std::string& path) const;

    SimParams params_{};
    MeshData mesh_;
    CsrAdjacency csr_;
    SubspaceBasisData basis_;

    std::vector<Vec3> x_rest_;
    std::vector<Vec3> x_;
    std::vector<Vec3> x_n_;
    std::vector<Vec3> v_n_;
    std::vector<Vec3> x_star_;

    std::vector<float> q_;
    std::vector<float> reduced_g_;

    std::vector<Tri> surface_tris_;
    VkTwoPhaseOps gpu_;
    TimingStats timing_stats_{};
};
