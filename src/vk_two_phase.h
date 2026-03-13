#pragma once

#include <string>
#include <vector>

#include "sim_params.h"
#include "types.h"
#include "vulkan_context.h"

class VkTwoPhaseOps {
public:
    struct CgSolveStats {
        std::uint32_t iterations = 0;
        std::uint32_t hessian_calls = 0;
    };

    bool Initialize(const MeshData& mesh,
                    const CsrAdjacency& csr,
                    const SubspaceBasisData& basis,
                    const std::vector<Vec3>& x_rest,
                    const SimParams& params,
                    std::string& error);
    void Shutdown();

    bool UploadState(const std::vector<Vec3>& x, const std::vector<Vec3>& x_star, std::string& error);
    bool PredictState(std::string& error);
    bool UpdateVelocityState(std::string& error);
    bool ReconstructXFromReduced(const std::vector<float>& q_reduced, std::string& error);
    bool DownloadX(std::vector<Vec3>& out_x, std::string& error);
    bool ComputeGradient(std::vector<Vec3>& out_force, std::string& error);
    bool ComputeHessianVec(const std::vector<Vec3>& p, std::vector<Vec3>& out_result, std::string& error);
    bool ComputeGradientReduced(std::vector<float>& out_reduced, std::string& error);
    bool ComputeGradientReducedFromQ(const std::vector<float>& q_reduced,
                                     std::vector<float>& out_reduced,
                                     std::string& error);
    bool ComputeHessianReduced(const std::vector<float>& p_reduced, std::vector<float>& out_reduced, std::string& error);
    bool SolveReducedCgGpuFixed(const std::vector<float>& rhs,
                                int cg_iters,
                                std::vector<float>& out_x,
                                CgSolveStats& out_stats,
                                std::string& error);
    bool ReconstructAndUpdateVelocityFromQ(const std::vector<float>& q_reduced, std::string& error);

    bool IsInitialized() const { return initialized_; }
    const std::string& DeviceName() const { return vk_.PhysicalDeviceName(); }

private:
    struct alignas(16) Vec4Std {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;
    };

    struct PushConstants {
        std::uint32_t num_verts = 0;
        std::uint32_t num_tets = 0;
        float inv_dt2 = 0.0f;
        float mu = 0.0f;
        float lam = 0.0f;
        float alpha = 0.0f;
        float ground_y = 0.0f;
        float ground_k = 0.0f;
        std::uint32_t use_ground = 0;
        std::uint32_t num_reduced = 0;
        float dt = 0.0f;
        float inv_dt = 0.0f;
        float gravity_x = 0.0f;
        float gravity_y = 0.0f;
        float gravity_z = 0.0f;
        float cg_damping = 0.0f;
        float cg_rel_tol = 0.0f;
        std::uint32_t cg_max_iters = 0;
        std::uint32_t _pad0 = 0;
    };

    struct ComputePipeline {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    };

    bool CreateComputePipeline(const std::string& spv_path,
                               const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                               ComputePipeline& out_pipeline,
                               std::string& error);
    void DestroyComputePipeline(ComputePipeline& pipeline);

    bool CreateAndUploadStaticBuffers(const MeshData& mesh,
                                      const CsrAdjacency& csr,
                                      const SubspaceBasisData& basis,
                                      const std::vector<Vec3>& x_rest,
                                      std::string& error);
    bool CreateDynamicBuffers(std::size_t num_verts,
                              std::size_t num_tets,
                              std::size_t num_reduced,
                              std::string& error);
    void DestroyBuffers();

    bool UpdateDescriptors(std::string& error);
    bool RunPredictDispatch(std::string& error);
    bool RunUpdateVelocityDispatch(std::string& error);
    bool RunGradientDispatch(std::string& error);
    bool RunHessianDispatch(std::string& error);
    bool RunReconstructXDispatch(std::string& error);
    bool RunGradientReducedDispatch(std::string& error);
    bool RunHessianReducedDispatch(std::string& error);
    bool RunReducedCgInitDispatch(std::string& error);
    bool RunReducedCgUpdateDispatch(std::string& error);

    void InsertComputeBarrier(VkCommandBuffer cmd) const;
    void InsertComputeToIndirectBarrier(VkCommandBuffer cmd) const;
    void RecordPredictDispatch(VkCommandBuffer cmd) const;
    void RecordUpdateVelocityDispatch(VkCommandBuffer cmd) const;
    void RecordGradientDispatch(VkCommandBuffer cmd) const;
    void RecordHessianDispatch(VkCommandBuffer cmd) const;
    void RecordReconstructXDispatch(VkCommandBuffer cmd) const;
    void RecordGradientReducedDispatch(VkCommandBuffer cmd) const;
    void RecordHessianReducedDispatch(VkCommandBuffer cmd) const;
    void RecordHessianReducedDispatchIndirect(VkCommandBuffer cmd) const;
    void RecordReducedCgInitDispatch(VkCommandBuffer cmd) const;
    void RecordReducedCgUpdateDispatch(VkCommandBuffer cmd) const;

    static std::vector<Vec4Std> ToVec4(const std::vector<Vec3>& v);
    static std::vector<Vec3> FromVec4(const std::vector<Vec4Std>& v);

    VulkanContext vk_;
    bool initialized_ = false;

    std::size_t num_verts_ = 0;
    std::size_t num_tets_ = 0;
    std::size_t num_reduced_ = 0;
    std::size_t num_project_chunks_ = 0;
    PushConstants push_{};

    VulkanContext::Buffer x_buf_;
    VulkanContext::Buffer x_star_buf_;
    VulkanContext::Buffer x_n_buf_;
    VulkanContext::Buffer v_n_buf_;
    VulkanContext::Buffer p_buf_;
    VulkanContext::Buffer force_buf_;
    VulkanContext::Buffer result_buf_;
    VulkanContext::Buffer tet_contrib_grad_buf_;
    VulkanContext::Buffer tet_contrib_hess_buf_;

    VulkanContext::Buffer basis_buf_;
    VulkanContext::Buffer x_rest_buf_;
    VulkanContext::Buffer mass_buf_;
    VulkanContext::Buffer tets_buf_;
    VulkanContext::Buffer dminv_buf_;
    VulkanContext::Buffer vol_buf_;
    VulkanContext::Buffer csr_offsets_buf_;
    VulkanContext::Buffer csr_tets_buf_;
    VulkanContext::Buffer csr_local_buf_;
    VulkanContext::Buffer reduced_in_buf_;
    VulkanContext::Buffer reduced_out_buf_;
    VulkanContext::Buffer reduced_q_buf_;
    VulkanContext::Buffer reduced_partial_buf_;
    VulkanContext::Buffer reduced_cg_r_buf_;
    VulkanContext::Buffer reduced_cg_x_buf_;
    VulkanContext::Buffer reduced_cg_ctrl_buf_;
    VulkanContext::Buffer reduced_cg_dispatch_buf_;

    ComputePipeline grad_stage_a_;
    ComputePipeline grad_stage_b_;
    ComputePipeline hess_stage_a_;
    ComputePipeline hess_stage_b_;
    ComputePipeline predict_state_;
    ComputePipeline update_velocity_state_;
    ComputePipeline reconstruct_x_;
    ComputePipeline build_world_;
    ComputePipeline project_force_stage1_;
    ComputePipeline project_result_stage1_;
    ComputePipeline project_stage2_;
    ComputePipeline reduced_cg_init_;
    ComputePipeline reduced_cg_update_;
};
