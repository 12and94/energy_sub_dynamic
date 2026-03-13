#pragma once

#include "types.h"

struct SimParams {
    float time_step = 1.0f / 60.0f;
    int num_substeps = 1;
    float dt = 0.0f;

    Vec3 gravity = Vec3{0.0f, -10.0f, 0.0f};
    int newton_iters = 2;
    int cg_iters = 8;
    float cg_rel_tol = 1e-2f;
    float convergence_reduced_avg_norm = 1e-8f;

    float rho = 1000.0f;
    float youngs_modulus = 100000.0f;
    float poissons_ratio = 0.30f;

    float mu = 0.0f;
    float lam = 0.0f;
    float alpha_gaia = 0.0f;
    float rest_energy_density = 0.0f;

    float reduced_damping = 1e-7f;
    bool use_reduced_direct_solve = false;
    int direct_solver_max_dim = 128;
    bool use_ground = true;
    float ground_y = -5.0f;
    float ground_k = 100000.0f;

    int subspace_dim_limit = 40;

    float initial_height = 0.0f;
    float initial_stretch_x = 1.0f;
    float initial_stretch_y = 1.0f;
    float initial_stretch_z = 1.0f;
};

inline void ComputeDerivedParams(SimParams& p) {
    p.mu = p.youngs_modulus / (2.0f * (1.0f + p.poissons_ratio));
    p.lam = p.youngs_modulus * p.poissons_ratio /
            ((1.0f + p.poissons_ratio) * (1.0f - 2.0f * p.poissons_ratio));
    p.alpha_gaia = 1.0f + p.mu / p.lam;
    const float one_minus = 1.0f - p.alpha_gaia;
    p.rest_energy_density = 0.5f * p.lam * one_minus * one_minus;
    p.dt = p.time_step / static_cast<float>(p.num_substeps);
}
