#pragma once

#include <array>
#include <cstdint>
#include <vector>

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& a, float s) {
    return Vec3{a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator*(float s, const Vec3& a) {
    return a * s;
}

inline Vec3 operator/(const Vec3& a, float s) {
    return Vec3{a.x / s, a.y / s, a.z / s};
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline Vec3& operator-=(Vec3& a, const Vec3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

struct Tet {
    std::uint32_t i0 = 0;
    std::uint32_t i1 = 0;
    std::uint32_t i2 = 0;
    std::uint32_t i3 = 0;
};

struct Tri {
    std::uint32_t i0 = 0;
    std::uint32_t i1 = 0;
    std::uint32_t i2 = 0;
};

struct MeshData {
    std::vector<Vec3> verts;
    std::vector<Tet> tets;
    std::vector<std::array<float, 9>> dm_inv;
    std::vector<float> vol_rest;
    std::vector<float> mass;
    std::vector<float> inv_mass;
};

struct CsrAdjacency {
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> tet_ids;
    std::vector<std::uint32_t> local_ids;
};

struct SubspaceBasisData {
    int rows = 0;
    int cols = 0;
    std::vector<float> basis_row_major;

    float youngs_modulus = 100000.0f;
    float poissons_ratio = 0.30f;
    bool has_material = false;

    std::vector<float> vertex_mass;
    bool has_vertex_mass = false;
};
