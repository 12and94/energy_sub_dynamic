#include "simple_renderer.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr float kDefaultYaw = 0.78539816339f;   // 45 deg
constexpr float kDefaultPitch = 0.78539816339f; // 45 deg
constexpr float kMinPitch = -1.45f;
constexpr float kMaxPitch = 1.45f;
constexpr float kNearClip = 0.02f;
constexpr float kRotateSpeed = 0.0085f;

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3f operator+(const Vec3f& a, const Vec3f& b) {
    return Vec3f{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3f operator-(const Vec3f& a, const Vec3f& b) {
    return Vec3f{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3f operator*(const Vec3f& a, float s) {
    return Vec3f{a.x * s, a.y * s, a.z * s};
}

float Dot(const Vec3f& a, const Vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3f Cross(const Vec3f& a, const Vec3f& b) {
    return Vec3f{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float Length(const Vec3f& v) {
    return std::sqrt(Dot(v, v));
}

Vec3f Normalize(const Vec3f& v) {
    const float len = Length(v);
    if (len < 1e-8f) {
        return Vec3f{0.0f, 0.0f, 0.0f};
    }
    return v * (1.0f / len);
}

Vec3f FromArray(const std::array<float, 3>& a) {
    return Vec3f{a[0], a[1], a[2]};
}

std::array<float, 3> ToArray(const Vec3f& v) {
    return {v.x, v.y, v.z};
}

float ComputeCameraDistance(const SimpleRenderer::Config& config, float radius) {
    const float half_fov_x = std::atan(static_cast<float>(config.width) * 0.5f / config.focal);
    const float min_dist = radius / std::max(std::tan(half_fov_x), 1e-4f);
    return std::max(min_dist * 2.0f, radius * 2.8f + 0.8f);
}

float ClampCameraDistance(float distance, float scene_radius) {
    const float min_dist = std::max(0.05f, scene_radius * 0.2f);
    const float max_dist = std::max(min_dist * 2.0f, scene_radius * 20.0f + 10.0f);
    return std::clamp(distance, min_dist, max_dist);
}

std::string SdlError(const std::string& prefix) {
    return prefix + ": " + SDL_GetError();
}

} // namespace

SimpleRenderer::~SimpleRenderer() {
    Shutdown();
}

void SimpleRenderer::ResetCamera() {
    camera_yaw_ = kDefaultYaw;
    camera_pitch_ = kDefaultPitch;
    camera_distance_ = 2.0f;
    camera_target_ = {0.0f, 0.0f, 0.0f};
    camera_scene_radius_ = 1.0f;
    camera_initialized_ = false;
    camera_dirty_ = false;
}

void SimpleRenderer::Shutdown() {
    if (texture_ != nullptr) {
        SDL_DestroyTexture(texture_);
        texture_ = nullptr;
    }
    if (renderer_ != nullptr) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }
    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    if (sdl_video_ready_) {
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        sdl_video_ready_ = false;
    }

    color_bgra_.clear();
    depth_.clear();
    initialized_ = false;
    window_closed_ = false;
    right_mouse_down_ = false;
    ResetCamera();
}

bool SimpleRenderer::Initialize(const Config& config, std::string& error) {
    Shutdown();
    if (config.width <= 0 || config.height <= 0 || config.focal <= 0.0f) {
        error = "Invalid SimpleRenderer config.";
        return false;
    }
    config_ = config;
    window_closed_ = false;
    color_bgra_.assign(static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(config_.height) * 4u, 0u);
    depth_.assign(static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(config_.height),
                  std::numeric_limits<float>::infinity());

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        error = SdlError("SDL_Init(SDL_INIT_VIDEO) failed");
        Shutdown();
        return false;
    }
    sdl_video_ready_ = true;

    window_ = SDL_CreateWindow(config_.window_title.c_str(),
                               config_.width,
                               config_.height,
                               SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (window_ == nullptr) {
        error = SdlError("SDL_CreateWindow failed");
        Shutdown();
        return false;
    }

    renderer_ = SDL_CreateRenderer(window_, nullptr);
    if (renderer_ == nullptr) {
        error = SdlError("SDL_CreateRenderer failed");
        Shutdown();
        return false;
    }

    texture_ = SDL_CreateTexture(renderer_,
                                 SDL_PIXELFORMAT_BGRA32,
                                 SDL_TEXTUREACCESS_STREAMING,
                                 config_.width,
                                 config_.height);
    if (texture_ == nullptr) {
        error = SdlError("SDL_CreateTexture failed");
        Shutdown();
        return false;
    }

    if (!SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255)) {
        error = SdlError("SDL_SetRenderDrawColor failed");
        Shutdown();
        return false;
    }

    ResetCamera();
    initialized_ = true;
    return true;
}

bool SimpleRenderer::PumpEvents() {
    if (window_ == nullptr) {
        return false;
    }
    const SDL_WindowID window_id = SDL_GetWindowID(window_);
    SDL_Event event{};
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
        case SDL_EVENT_QUIT:
            window_closed_ = true;
            return false;
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            if (event.window.windowID == window_id) {
                window_closed_ = true;
                return false;
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event.button.windowID == window_id && event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse_down_ = true;
                (void)SDL_SetWindowRelativeMouseMode(window_, true);
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.windowID == window_id && event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse_down_ = false;
                (void)SDL_SetWindowRelativeMouseMode(window_, false);
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (event.motion.windowID == window_id && (right_mouse_down_ || SDL_GetWindowRelativeMouseMode(window_))) {
                camera_yaw_ -= event.motion.xrel * kRotateSpeed;
                camera_pitch_ += event.motion.yrel * kRotateSpeed;
                camera_pitch_ = std::clamp(camera_pitch_, kMinPitch, kMaxPitch);
                camera_dirty_ = true;
            }
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            if (event.wheel.windowID == window_id) {
                float wheel = event.wheel.y;
                if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) {
                    wheel = -wheel;
                }
                if (std::abs(wheel) > 1e-6f) {
                    camera_distance_ = ClampCameraDistance(camera_distance_ * std::pow(0.88f, wheel), camera_scene_radius_);
                    camera_dirty_ = true;
                }
            }
            break;
        case SDL_EVENT_KEY_DOWN:
            if (event.key.windowID != window_id || event.key.repeat) {
                break;
            }
            if (event.key.key == SDLK_ESCAPE) {
                window_closed_ = true;
                return false;
            }
            if (event.key.key == SDLK_R) {
                ResetCamera();
                break;
            }
            break;
        default:
            break;
        }
    }
    return !window_closed_;
}

void SimpleRenderer::ClearFrameBuffer() {
    std::fill(depth_.begin(), depth_.end(), std::numeric_limits<float>::infinity());
    const int w = config_.width;
    const int h = config_.height;
    for (int y = 0; y < h; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(std::max(1, h - 1));
        const std::uint8_t r = static_cast<std::uint8_t>(18.0f + 30.0f * t);
        const std::uint8_t g = static_cast<std::uint8_t>(22.0f + 40.0f * t);
        const std::uint8_t b = static_cast<std::uint8_t>(28.0f + 60.0f * t);
        for (int x = 0; x < w; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y * w + x) * 4u;
            color_bgra_[idx + 0] = b;
            color_bgra_[idx + 1] = g;
            color_bgra_[idx + 2] = r;
            color_bgra_[idx + 3] = 255u;
        }
    }
}

bool SimpleRenderer::Present(std::string& error) {
    if (window_closed_) {
        return true;
    }
    if (!SDL_UpdateTexture(texture_, nullptr, color_bgra_.data(), config_.width * 4)) {
        error = SdlError("SDL_UpdateTexture failed");
        return false;
    }
    if (!SDL_RenderClear(renderer_)) {
        error = SdlError("SDL_RenderClear failed");
        return false;
    }
    if (!SDL_RenderTexture(renderer_, texture_, nullptr, nullptr)) {
        error = SdlError("SDL_RenderTexture failed");
        return false;
    }
    if (!SDL_RenderPresent(renderer_)) {
        error = SdlError("SDL_RenderPresent failed");
        return false;
    }
    return true;
}

float SimpleRenderer::EstimateMeshRadius(const std::vector<Vec3>& vertices) const {
    if (vertices.empty()) {
        return 1.0f;
    }
    Vec3f bmin{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    Vec3f bmax{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };
    for (const Vec3& v : vertices) {
        bmin.x = std::min(bmin.x, v.x);
        bmin.y = std::min(bmin.y, v.y);
        bmin.z = std::min(bmin.z, v.z);
        bmax.x = std::max(bmax.x, v.x);
        bmax.y = std::max(bmax.y, v.y);
        bmax.z = std::max(bmax.z, v.z);
    }
    const Vec3f ext{
        0.5f * std::max(1e-4f, bmax.x - bmin.x),
        0.5f * std::max(1e-4f, bmax.y - bmin.y),
        0.5f * std::max(1e-4f, bmax.z - bmin.z)
    };
    return std::max(Length(ext), 1e-3f);
}

void SimpleRenderer::UpdateCameraFromMeshBounds(const std::vector<Vec3>& vertices) {
    if (vertices.empty()) {
        return;
    }

    Vec3f bmin{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    Vec3f bmax{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };
    for (const Vec3& v : vertices) {
        bmin.x = std::min(bmin.x, v.x);
        bmin.y = std::min(bmin.y, v.y);
        bmin.z = std::min(bmin.z, v.z);
        bmax.x = std::max(bmax.x, v.x);
        bmax.y = std::max(bmax.y, v.y);
        bmax.z = std::max(bmax.z, v.z);
    }

    const Vec3f center{
        0.5f * (bmin.x + bmax.x),
        0.5f * (bmin.y + bmax.y),
        0.5f * (bmin.z + bmax.z)
    };
    const float radius = EstimateMeshRadius(vertices);

    if (!camera_initialized_) {
        camera_target_ = ToArray(center);
        camera_scene_radius_ = radius;
        camera_distance_ = ClampCameraDistance(ComputeCameraDistance(config_, radius), radius);
        camera_initialized_ = true;
        return;
    }

    // Auto-follow the mesh center only until user interacts with the camera.
    if (!camera_dirty_) {
        camera_target_ = ToArray(center);
        camera_scene_radius_ = radius;
        camera_distance_ = ClampCameraDistance(ComputeCameraDistance(config_, radius), radius);
        return;
    }

    camera_scene_radius_ = std::max(camera_scene_radius_, radius);
    camera_distance_ = ClampCameraDistance(camera_distance_, camera_scene_radius_);
}

void SimpleRenderer::BuildCameraBasis(std::array<float, 3>& camera_pos,
                                      std::array<float, 3>& camera_target,
                                      std::array<float, 3>& camera_right,
                                      std::array<float, 3>& camera_up,
                                      std::array<float, 3>& camera_forward) const {
    const Vec3f target = FromArray(camera_target_);

    const float cp = std::cos(camera_pitch_);
    const float sp = std::sin(camera_pitch_);
    const float cy = std::cos(camera_yaw_);
    const float sy = std::sin(camera_yaw_);
    const Vec3f orbit = Normalize(Vec3f{
        cp * cy,
        sp,
        cp * sy
    });
    const Vec3f pos = target + orbit * camera_distance_;
    const Vec3f forward = Normalize(target - pos);
    const Vec3f world_up{0.0f, 1.0f, 0.0f};
    Vec3f right = Normalize(Cross(forward, world_up));
    if (Length(right) < 1e-6f) {
        right = Vec3f{1.0f, 0.0f, 0.0f};
    }
    const Vec3f up = Normalize(Cross(right, forward));

    camera_pos = ToArray(pos);
    camera_target = ToArray(target);
    camera_right = ToArray(right);
    camera_up = ToArray(up);
    camera_forward = ToArray(forward);
}

bool SimpleRenderer::RenderFrame(const std::vector<Vec3>& vertices,
                                 const std::vector<Tri>& triangles,
                                 int frame_index,
                                 std::string& error) {
    (void)frame_index;
    if (!initialized_) {
        error = "SimpleRenderer is not initialized.";
        return false;
    }
    if (!PumpEvents()) {
        return true;
    }
    if (vertices.empty() || triangles.empty()) {
        error = "SimpleRenderer input mesh is empty.";
        return false;
    }

    UpdateCameraFromMeshBounds(vertices);
    ClearFrameBuffer();

    std::array<float, 3> cam_pos_arr{};
    std::array<float, 3> cam_target_arr{};
    std::array<float, 3> cam_right_arr{};
    std::array<float, 3> cam_up_arr{};
    std::array<float, 3> cam_forward_arr{};
    BuildCameraBasis(cam_pos_arr, cam_target_arr, cam_right_arr, cam_up_arr, cam_forward_arr);
    const Vec3f cam_pos = FromArray(cam_pos_arr);
    const Vec3f cam_right = FromArray(cam_right_arr);
    const Vec3f cam_up = FromArray(cam_up_arr);
    const Vec3f cam_forward = FromArray(cam_forward_arr);

    // Keep key light near camera direction so shaded relief is easy to read.
    const Vec3f light_dir = Normalize(cam_forward + cam_up * 0.20f + cam_right * 0.08f);

    const int w = config_.width;
    const int h = config_.height;

    struct ProjectedV {
        float x = 0.0f;
        float y = 0.0f;
        float z = -1.0f;
        bool valid = false;
        Vec3f world{};
    };
    std::vector<ProjectedV> proj(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const Vec3f p{vertices[i].x, vertices[i].y, vertices[i].z};
        const Vec3f rel = p - cam_pos;
        const float cx = Dot(rel, cam_right);
        const float cy = Dot(rel, cam_up);
        const float cz = Dot(rel, cam_forward);
        if (cz <= kNearClip) {
            continue;
        }
        const float inv_z = 1.0f / cz;
        const float sx = static_cast<float>(w) * 0.5f + config_.focal * cx * inv_z;
        const float sy = static_cast<float>(h) * 0.5f - config_.focal * cy * inv_z;
        proj[i] = ProjectedV{sx, sy, cz, true, p};
    }

    for (const Tri& tri : triangles) {
        if (tri.i0 >= proj.size() || tri.i1 >= proj.size() || tri.i2 >= proj.size()) {
            continue;
        }
        const ProjectedV& v0 = proj[tri.i0];
        const ProjectedV& v1 = proj[tri.i1];
        const ProjectedV& v2 = proj[tri.i2];
        if (!v0.valid || !v1.valid || !v2.valid) {
            continue;
        }

        const float ax = v0.x;
        const float ay = v0.y;
        const float bx = v1.x;
        const float by = v1.y;
        const float cx = v2.x;
        const float cy = v2.y;
        const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (std::abs(area) < 1e-7f || area < 0.0f) {
            continue;
        }

        const int min_x = std::max(0, static_cast<int>(std::floor(std::min({ax, bx, cx}))));
        const int max_x = std::min(w - 1, static_cast<int>(std::ceil(std::max({ax, bx, cx}))));
        const int min_y = std::max(0, static_cast<int>(std::floor(std::min({ay, by, cy}))));
        const int max_y = std::min(h - 1, static_cast<int>(std::ceil(std::max({ay, by, cy}))));
        if (min_x > max_x || min_y > max_y) {
            continue;
        }

        const Vec3f n = Normalize(Cross(v1.world - v0.world, v2.world - v0.world));
        const float nl = std::clamp(Dot(n, light_dir), 0.0f, 1.0f);
        const float shade = 0.25f + 0.75f * nl;
        const std::uint8_t base_r = static_cast<std::uint8_t>(220.0f * shade);
        const std::uint8_t base_g = static_cast<std::uint8_t>(190.0f * shade);
        const std::uint8_t base_b = static_cast<std::uint8_t>(120.0f * shade);

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;
                const float fy = static_cast<float>(py) + 0.5f;
                const float w0 = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) / area;
                const float w1 = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) / area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                    continue;
                }

                const float z = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                const std::size_t id = static_cast<std::size_t>(py * w + px);
                if (z >= depth_[id]) {
                    continue;
                }
                depth_[id] = z;

                const std::size_t c = id * 4u;
                color_bgra_[c + 0] = base_b;
                color_bgra_[c + 1] = base_g;
                color_bgra_[c + 2] = base_r;
                color_bgra_[c + 3] = 255u;
            }
        }
    }

    return Present(error);
}
