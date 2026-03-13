#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "types.h"

struct SDL_Window;
struct SDL_Renderer;
struct SDL_Texture;

class SimpleRenderer {
public:
    struct Config {
        int width = 960;
        int height = 720;
        float focal = 760.0f;
        std::string window_title = "vk_cs realtime render";
    };

    ~SimpleRenderer();

    bool Initialize(const Config& config, std::string& error);
    bool RenderFrame(const std::vector<Vec3>& vertices,
                     const std::vector<Tri>& triangles,
                     int frame_index,
                     std::string& error);
    bool IsWindowClosed() const { return window_closed_; }
    void Shutdown();

private:
    void ClearFrameBuffer();
    bool PumpEvents();
    bool Present(std::string& error);
    void ResetCamera();
    void UpdateCameraFromMeshBounds(const std::vector<Vec3>& vertices);
    void BuildCameraBasis(std::array<float, 3>& camera_pos,
                          std::array<float, 3>& camera_target,
                          std::array<float, 3>& camera_right,
                          std::array<float, 3>& camera_up,
                          std::array<float, 3>& camera_forward) const;
    float EstimateMeshRadius(const std::vector<Vec3>& vertices) const;

    Config config_{};
    bool initialized_ = false;
    bool window_closed_ = false;
    bool sdl_video_ready_ = false;
    bool camera_initialized_ = false;
    bool camera_dirty_ = false;
    bool right_mouse_down_ = false;
    std::vector<std::uint8_t> color_bgra_;
    std::vector<float> depth_;
    std::array<float, 3> camera_target_{0.0f, 0.0f, 0.0f};
    float camera_yaw_ = 0.78539816339f;
    float camera_pitch_ = 0.78539816339f;
    float camera_distance_ = 2.0f;
    float camera_scene_radius_ = 1.0f;

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* texture_ = nullptr;
};
