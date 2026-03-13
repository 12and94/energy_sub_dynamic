#include "subspace_simulator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "csr_builder.h"
#include "mesh_loader.h"
#include "subspace_loader.h"
#include "simple_renderer.h"

namespace {
using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

Vec3 RotateAroundZ(const Vec3& p, float theta) {
    const float c = std::cos(theta);
    const float s = std::sin(theta);
    return Vec3{
        c * p.x - s * p.y,
        s * p.x + c * p.y,
        p.z
    };
}

struct FaceKey {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::uint32_t c = 0;
    bool operator==(const FaceKey& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& k) const {
        const std::size_t h1 = std::hash<std::uint32_t>{}(k.a);
        const std::size_t h2 = std::hash<std::uint32_t>{}(k.b);
        const std::size_t h3 = std::hash<std::uint32_t>{}(k.c);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

float DotReduced(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

bool CholeskySolve(std::vector<float>& a, std::vector<float>& b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = a[static_cast<std::size_t>(i * n + j)];
            for (int k = 0; k < j; ++k) {
                sum -= a[static_cast<std::size_t>(i * n + k)] * a[static_cast<std::size_t>(j * n + k)];
            }
            if (i == j) {
                if (sum <= 1e-10f) {
                    return false;
                }
                a[static_cast<std::size_t>(i * n + i)] = std::sqrt(sum);
            } else {
                a[static_cast<std::size_t>(i * n + j)] = sum / a[static_cast<std::size_t>(j * n + j)];
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        float sum = b[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k) {
            sum -= a[static_cast<std::size_t>(i * n + k)] * b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] = sum / a[static_cast<std::size_t>(i * n + i)];
    }

    for (int i = n - 1; i >= 0; --i) {
        float sum = b[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k) {
            sum -= a[static_cast<std::size_t>(k * n + i)] * b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] = sum / a[static_cast<std::size_t>(i * n + i)];
    }
    return true;
}

} // namespace

bool SubspaceSimulator::Initialize(const std::string& mesh_path,
                                   const std::string& basis_path,
                                   SimParams params,
                                   std::string& error) {
    params_ = params;

    if (!LoadTetMeshAuto(mesh_path, mesh_, error)) {
        return false;
    }
    if (!LoadSubspaceBasisJson(basis_path, params_.subspace_dim_limit, basis_, error)) {
        return false;
    }
    if (basis_.rows != static_cast<int>(mesh_.verts.size() * 3)) {
        error = "Basis row count does not match mesh DOF.";
        return false;
    }

    if (basis_.has_material) {
        params_.youngs_modulus = basis_.youngs_modulus;
        params_.poissons_ratio = basis_.poissons_ratio;
    }
    ComputeDerivedParams(params_);

    const std::size_t n = mesh_.verts.size();
    x_rest_.assign(n, Vec3{});
    x_.assign(n, Vec3{});
    x_n_.assign(n, Vec3{});
    v_n_.assign(n, Vec3{0.0f, 0.0f, 0.0f});
    x_star_.assign(n, Vec3{});

    const float theta = 0.5f;
    const float sx = params_.initial_stretch_x;
    const float sy = params_.initial_stretch_y;
    const float sz = params_.initial_stretch_z;

    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 v_orig = mesh_.verts[i] - Vec3{0.5f, 0.5f, 0.5f};
        const Vec3 v_rot = RotateAroundZ(v_orig, theta);
        const Vec3 v_base = v_rot + Vec3{0.5f, params_.initial_height, 0.5f};
        x_rest_[i] = v_base;

        Vec3 v_stretched = v_base;
        v_stretched.x = 0.5f + (v_base.x - 0.5f) * sx;
        v_stretched.y = params_.initial_height + (v_base.y - params_.initial_height) * sy;
        v_stretched.z = 0.5f + (v_base.z - 0.5f) * sz;
        x_[i] = v_stretched;
        x_n_[i] = v_stretched;
        x_star_[i] = v_stretched;
    }

    if (!BuildRestTetData(x_rest_, params_.rho, mesh_, error)) {
        return false;
    }
    if (basis_.has_vertex_mass && basis_.vertex_mass.size() == n) {
        mesh_.mass = basis_.vertex_mass;
        mesh_.inv_mass.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            mesh_.inv_mass[i] = mesh_.mass[i] > 0.0f ? 1.0f / mesh_.mass[i] : 0.0f;
        }
    }

    csr_ = BuildTetVertexCsr(mesh_.verts.size(), mesh_.tets);
    q_.assign(static_cast<std::size_t>(basis_.cols), 0.0f);
    reduced_g_.assign(static_cast<std::size_t>(basis_.cols), 0.0f);

    if (!BuildSurfaceTriangles()) {
        error = "Failed to build surface triangle list.";
        return false;
    }

    if (!gpu_.Initialize(mesh_, csr_, basis_, x_rest_, params_, error)) {
        return false;
    }
    if (!gpu_.UploadState(x_, x_star_, error)) {
        return false;
    }
    return true;
}

bool SubspaceSimulator::Substep(std::string& error) {
    auto t0 = Clock::now();
    if (!gpu_.PredictState(error)) {
        return false;
    }
    timing_stats_.predict_ms += ElapsedMs(t0, Clock::now());

    const int r = basis_.cols;
    if (r <= 0) {
        error = "Subspace dimension is zero.";
        return false;
    }

    for (int it = 0; it < params_.newton_iters; ++it) {
        timing_stats_.newton_iterations += 1;

        t0 = Clock::now();
        if (!gpu_.ComputeGradientReducedFromQ(q_, reduced_g_, error)) {
            return false;
        }
        timing_stats_.gradient_ms += ElapsedMs(t0, Clock::now());
        const float reduced_norm = std::sqrt(DotReduced(reduced_g_, reduced_g_)) / static_cast<float>(r);
        if (reduced_norm < params_.convergence_reduced_avg_norm) {
            break;
        }

        std::vector<float> rhs(static_cast<std::size_t>(r), 0.0f);
        for (int i = 0; i < r; ++i) {
            rhs[static_cast<std::size_t>(i)] = -reduced_g_[static_cast<std::size_t>(i)];
        }

        std::vector<float> xk(static_cast<std::size_t>(r), 0.0f);
        bool solved = false;
        const auto solve_begin = Clock::now();

        if (params_.use_reduced_direct_solve && r <= params_.direct_solver_max_dim) {
            std::vector<float> h_red(static_cast<std::size_t>(r * r), 0.0f);
            std::vector<float> unit_col(static_cast<std::size_t>(r), 0.0f);
            std::vector<float> y_col;

            for (int col = 0; col < r; ++col) {
                std::fill(unit_col.begin(), unit_col.end(), 0.0f);
                unit_col[static_cast<std::size_t>(col)] = 1.0f;
                const auto h_begin = Clock::now();
                if (!gpu_.ComputeHessianReduced(unit_col, y_col, error)) {
                    return false;
                }
                timing_stats_.hessian_ms += ElapsedMs(h_begin, Clock::now());
                timing_stats_.hessian_calls += 1;
                for (int row = 0; row < r; ++row) {
                    h_red[static_cast<std::size_t>(row * r + col)] = y_col[static_cast<std::size_t>(row)];
                }
            }

            for (int i = 0; i < r; ++i) {
                for (int j = i + 1; j < r; ++j) {
                    const float sym = 0.5f * (h_red[static_cast<std::size_t>(i * r + j)] +
                                              h_red[static_cast<std::size_t>(j * r + i)]);
                    h_red[static_cast<std::size_t>(i * r + j)] = sym;
                    h_red[static_cast<std::size_t>(j * r + i)] = sym;
                }
            }

            std::vector<float> rhs_try = rhs;
            std::vector<float> h_try = h_red;
            float reg = params_.reduced_damping;
            for (int attempt = 0; attempt < 4; ++attempt) {
                h_try = h_red;
                rhs_try = rhs;
                for (int d = 0; d < r; ++d) {
                    h_try[static_cast<std::size_t>(d * r + d)] += reg;
                }
                if (CholeskySolve(h_try, rhs_try, r)) {
                    xk = rhs_try;
                    solved = true;
                    break;
                }
                reg = std::max(reg * 10.0f, 1e-8f);
            }
        }

        if (!solved) {
            VkTwoPhaseOps::CgSolveStats cg_stats{};
            const auto cg_begin = Clock::now();
            if (!gpu_.SolveReducedCgGpuFixed(rhs, params_.cg_iters, xk, cg_stats, error)) {
                return false;
            }
            timing_stats_.hessian_ms += ElapsedMs(cg_begin, Clock::now());
            timing_stats_.cg_iterations += cg_stats.iterations;
            timing_stats_.hessian_calls += cg_stats.hessian_calls;
        }
        timing_stats_.linear_solve_ms += ElapsedMs(solve_begin, Clock::now());

        for (int k = 0; k < r; ++k) {
            q_[static_cast<std::size_t>(k)] += xk[static_cast<std::size_t>(k)];
        }
    }

    t0 = Clock::now();
    if (!gpu_.ReconstructAndUpdateVelocityFromQ(q_, error)) {
        return false;
    }
    timing_stats_.reconstruct_final_ms += ElapsedMs(t0, Clock::now());
    timing_stats_.substeps += 1;
    return true;
}

bool SubspaceSimulator::BuildSurfaceTriangles() {
    struct FaceData {
        int count = 0;
        Tri tri{};
    };

    std::unordered_map<FaceKey, FaceData, FaceKeyHash> face_map;
    face_map.reserve(mesh_.tets.size() * 4);

    for (const Tet& t : mesh_.tets) {
        const Tri faces[4] = {
            Tri{t.i0, t.i2, t.i1},
            Tri{t.i0, t.i1, t.i3},
            Tri{t.i0, t.i3, t.i2},
            Tri{t.i1, t.i2, t.i3}
        };
        for (const Tri& f : faces) {
            std::array<std::uint32_t, 3> idx{f.i0, f.i1, f.i2};
            std::sort(idx.begin(), idx.end());
            const FaceKey key{idx[0], idx[1], idx[2]};
            auto& data = face_map[key];
            data.count += 1;
            data.tri = f;
        }
    }

    surface_tris_.clear();
    for (const auto& kv : face_map) {
        if (kv.second.count == 1) {
            surface_tris_.push_back(kv.second.tri);
        }
    }
    return !surface_tris_.empty();
}

bool SubspaceSimulator::WriteObj(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    for (const Vec3& p : x_) {
        out << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }
    for (const Tri& t : surface_tris_) {
        out << "f " << (t.i0 + 1) << " " << (t.i1 + 1) << " " << (t.i2 + 1) << "\n";
    }
    return true;
}

bool SubspaceSimulator::Run(int total_frames,
                            const std::string& output_dir,
                            bool download_frames,
                            bool write_obj_frames,
                            bool render_frames,
                            std::string& error) {
    timing_stats_ = TimingStats{};
    if (write_obj_frames) {
        std::filesystem::create_directories(output_dir);
    }
    SimpleRenderer renderer;
    if (render_frames) {
        SimpleRenderer::Config cfg{};
        if (!renderer.Initialize(cfg, error)) {
            return false;
        }
    }
    for (int frame = 0; frame < total_frames; ++frame) {
        for (int s = 0; s < params_.num_substeps; ++s) {
            if (!Substep(error)) {
                return false;
            }
        }
        if (download_frames || write_obj_frames) {
            const auto t_download = Clock::now();
            if (!gpu_.DownloadX(x_, error)) {
                return false;
            }
            timing_stats_.download_x_ms += ElapsedMs(t_download, Clock::now());
        }

        if (write_obj_frames) {
            const std::string frame_path = output_dir + "/frame_" + std::to_string(frame) + ".obj";
            const auto t0 = Clock::now();
            if (!WriteObj(frame_path)) {
                error = "Failed to write frame: " + frame_path;
                return false;
            }
            timing_stats_.write_obj_ms += ElapsedMs(t0, Clock::now());
        }
        if (render_frames) {
            const auto t_render = Clock::now();
            if (!renderer.RenderFrame(x_, surface_tris_, frame, error)) {
                return false;
            }
            timing_stats_.render_ms += ElapsedMs(t_render, Clock::now());
            if (renderer.IsWindowClosed()) {
                std::cout << "Render window closed, stopping simulation loop.\n";
                break;
            }
        }
        std::cout << "Frame " << frame << " done\n";
    }
    return true;
}
