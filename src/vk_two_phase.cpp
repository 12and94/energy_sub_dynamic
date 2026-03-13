#include "vk_two_phase.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>

namespace {

constexpr std::uint32_t kTetVertWorkgroupSize = 256;
constexpr std::uint32_t kProjectWorkgroupSize = 64;

struct alignas(16) CgControlStd {
    float rr = 0.0f;
    float rr_stop = 0.0f;
    std::uint32_t iter = 0;
    std::uint32_t converged = 0;
};

struct CgDispatchArgsStd {
    VkDispatchIndirectCommand build_world{};
    VkDispatchIndirectCommand hess_tet{};
    VkDispatchIndirectCommand hess_vert{};
    VkDispatchIndirectCommand project_stage1{};
    VkDispatchIndirectCommand project_stage2{};
};

static_assert(sizeof(CgDispatchArgsStd) == sizeof(VkDispatchIndirectCommand) * 5,
              "Unexpected indirect dispatch buffer layout.");

VkDescriptorSetLayoutBinding StorageBinding(std::uint32_t binding) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

} // namespace

bool VkTwoPhaseOps::CreateComputePipeline(const std::string& spv_path,
                                          const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                          ComputePipeline& out_pipeline,
                                          std::string& error) {
    out_pipeline = ComputePipeline{};
    VkShaderModule shader = VK_NULL_HANDLE;
    if (!vk_.CreateShaderModuleFromFile(spv_path, shader, error)) {
        return false;
    }

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    set_layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(vk_.Device(), &set_layout_info, nullptr, &out_pipeline.set_layout) != VK_SUCCESS) {
        error = "vkCreateDescriptorSetLayout failed.";
        vkDestroyShaderModule(vk_.Device(), shader, nullptr);
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &out_pipeline.set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    if (vkCreatePipelineLayout(vk_.Device(), &layout_info, nullptr, &out_pipeline.pipeline_layout) != VK_SUCCESS) {
        error = "vkCreatePipelineLayout failed.";
        vkDestroyDescriptorSetLayout(vk_.Device(), out_pipeline.set_layout, nullptr);
        out_pipeline.set_layout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vk_.Device(), shader, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = out_pipeline.pipeline_layout;
    if (vkCreateComputePipelines(vk_.Device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &out_pipeline.pipeline) != VK_SUCCESS) {
        error = "vkCreateComputePipelines failed.";
        vkDestroyPipelineLayout(vk_.Device(), out_pipeline.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(vk_.Device(), out_pipeline.set_layout, nullptr);
        out_pipeline.pipeline_layout = VK_NULL_HANDLE;
        out_pipeline.set_layout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vk_.Device(), shader, nullptr);
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<std::uint32_t>(bindings.size());

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(vk_.Device(), &pool_info, nullptr, &out_pipeline.descriptor_pool) != VK_SUCCESS) {
        error = "vkCreateDescriptorPool failed.";
        vkDestroyPipeline(vk_.Device(), out_pipeline.pipeline, nullptr);
        vkDestroyPipelineLayout(vk_.Device(), out_pipeline.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(vk_.Device(), out_pipeline.set_layout, nullptr);
        out_pipeline.pipeline = VK_NULL_HANDLE;
        out_pipeline.pipeline_layout = VK_NULL_HANDLE;
        out_pipeline.set_layout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vk_.Device(), shader, nullptr);
        return false;
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = out_pipeline.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &out_pipeline.set_layout;
    if (vkAllocateDescriptorSets(vk_.Device(), &alloc_info, &out_pipeline.descriptor_set) != VK_SUCCESS) {
        error = "vkAllocateDescriptorSets failed.";
        vkDestroyDescriptorPool(vk_.Device(), out_pipeline.descriptor_pool, nullptr);
        vkDestroyPipeline(vk_.Device(), out_pipeline.pipeline, nullptr);
        vkDestroyPipelineLayout(vk_.Device(), out_pipeline.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(vk_.Device(), out_pipeline.set_layout, nullptr);
        out_pipeline.descriptor_pool = VK_NULL_HANDLE;
        out_pipeline.pipeline = VK_NULL_HANDLE;
        out_pipeline.pipeline_layout = VK_NULL_HANDLE;
        out_pipeline.set_layout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vk_.Device(), shader, nullptr);
        return false;
    }

    vkDestroyShaderModule(vk_.Device(), shader, nullptr);
    return true;
}

void VkTwoPhaseOps::DestroyComputePipeline(ComputePipeline& pipeline) {
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vk_.Device(), pipeline.descriptor_pool, nullptr);
        pipeline.descriptor_pool = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vk_.Device(), pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vk_.Device(), pipeline.pipeline_layout, nullptr);
        pipeline.pipeline_layout = VK_NULL_HANDLE;
    }
    if (pipeline.set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vk_.Device(), pipeline.set_layout, nullptr);
        pipeline.set_layout = VK_NULL_HANDLE;
    }
}

bool VkTwoPhaseOps::CreateAndUploadStaticBuffers(const MeshData& mesh,
                                                 const CsrAdjacency& csr,
                                                 const SubspaceBasisData& basis,
                                                 const std::vector<Vec3>& x_rest,
                                                 std::string& error) {
    const VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    const auto create_static = [&](VulkanContext::Buffer& b, VkDeviceSize size) -> bool {
        return vk_.CreateBuffer(size, storage_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b, error);
    };

    if (!create_static(basis_buf_, basis.basis_row_major.size() * sizeof(float))) return false;
    if (!create_static(x_rest_buf_, x_rest.size() * sizeof(Vec4Std))) return false;
    if (!create_static(mass_buf_, mesh.mass.size() * sizeof(float))) return false;
    if (!create_static(tets_buf_, mesh.tets.size() * sizeof(std::uint32_t) * 4)) return false;
    if (!create_static(dminv_buf_, mesh.dm_inv.size() * sizeof(float) * 9)) return false;
    if (!create_static(vol_buf_, mesh.vol_rest.size() * sizeof(float))) return false;
    if (!create_static(csr_offsets_buf_, csr.offsets.size() * sizeof(std::uint32_t))) return false;
    if (!create_static(csr_tets_buf_, csr.tet_ids.size() * sizeof(std::uint32_t))) return false;
    if (!create_static(csr_local_buf_, csr.local_ids.size() * sizeof(std::uint32_t))) return false;

    std::vector<std::uint32_t> tets_flat(mesh.tets.size() * 4);
    for (std::size_t i = 0; i < mesh.tets.size(); ++i) {
        tets_flat[i * 4 + 0] = mesh.tets[i].i0;
        tets_flat[i * 4 + 1] = mesh.tets[i].i1;
        tets_flat[i * 4 + 2] = mesh.tets[i].i2;
        tets_flat[i * 4 + 3] = mesh.tets[i].i3;
    }

    std::vector<float> dminv_flat(mesh.dm_inv.size() * 9);
    for (std::size_t i = 0; i < mesh.dm_inv.size(); ++i) {
        for (int k = 0; k < 9; ++k) {
            dminv_flat[i * 9 + static_cast<std::size_t>(k)] = mesh.dm_inv[i][static_cast<std::size_t>(k)];
        }
    }

    const std::vector<Vec4Std> x_rest4 = ToVec4(x_rest);

    if (!vk_.UploadToBuffer(basis_buf_, basis.basis_row_major.data(), basis_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(x_rest_buf_, x_rest4.data(), x_rest_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(mass_buf_, mesh.mass.data(), mass_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(tets_buf_, tets_flat.data(), tets_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(dminv_buf_, dminv_flat.data(), dminv_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(vol_buf_, mesh.vol_rest.data(), vol_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(csr_offsets_buf_, csr.offsets.data(), csr_offsets_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(csr_tets_buf_, csr.tet_ids.data(), csr_tets_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(csr_local_buf_, csr.local_ids.data(), csr_local_buf_.size, error)) return false;
    return true;
}

bool VkTwoPhaseOps::CreateDynamicBuffers(std::size_t num_verts,
                                         std::size_t num_tets,
                                         std::size_t num_reduced,
                                         std::string& error) {
    const VkDeviceSize vec_bytes = static_cast<VkDeviceSize>(num_verts * sizeof(Vec4Std));
    const VkDeviceSize tet_contrib_bytes = static_cast<VkDeviceSize>(num_tets * 4 * sizeof(Vec4Std));
    const VkDeviceSize reduced_bytes = static_cast<VkDeviceSize>(std::max<std::size_t>(num_reduced, 1) * sizeof(float));
    const VkDeviceSize partial_bytes =
        static_cast<VkDeviceSize>(std::max<std::size_t>(num_reduced * num_project_chunks_, 1) * sizeof(float));
    const VkDeviceSize cg_ctrl_bytes = static_cast<VkDeviceSize>(sizeof(CgControlStd));
    const VkDeviceSize cg_dispatch_bytes = static_cast<VkDeviceSize>(sizeof(CgDispatchArgsStd));
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, x_buf_, error)) return false;
    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, x_star_buf_, error)) return false;
    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, x_n_buf_, error)) return false;
    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, v_n_buf_, error)) return false;
    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, p_buf_, error)) return false;
    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, force_buf_, error)) return false;
    if (!vk_.CreateBuffer(vec_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, result_buf_, error)) return false;
    if (!vk_.CreateBuffer(tet_contrib_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tet_contrib_grad_buf_, error)) return false;
    if (!vk_.CreateBuffer(tet_contrib_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tet_contrib_hess_buf_, error)) return false;
    if (!vk_.CreateBuffer(reduced_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          reduced_in_buf_,
                          error)) {
        return false;
    }
    if (!vk_.CreateBuffer(reduced_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          reduced_out_buf_,
                          error)) {
        return false;
    }
    if (!vk_.CreateBuffer(reduced_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          reduced_q_buf_,
                          error)) {
        return false;
    }
    if (!vk_.CreateBuffer(reduced_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, reduced_cg_r_buf_, error)) return false;
    if (!vk_.CreateBuffer(reduced_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          reduced_cg_x_buf_,
                          error)) {
        return false;
    }
    if (!vk_.CreateBuffer(cg_ctrl_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          reduced_cg_ctrl_buf_,
                          error)) {
        return false;
    }
    if (!vk_.CreateBuffer(cg_dispatch_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          reduced_cg_dispatch_buf_,
                          error)) {
        return false;
    }
    if (!vk_.CreateBuffer(partial_bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, reduced_partial_buf_, error)) return false;
    return true;
}

void VkTwoPhaseOps::DestroyBuffers() {
    vk_.DestroyBuffer(x_buf_);
    vk_.DestroyBuffer(x_star_buf_);
    vk_.DestroyBuffer(x_n_buf_);
    vk_.DestroyBuffer(v_n_buf_);
    vk_.DestroyBuffer(p_buf_);
    vk_.DestroyBuffer(force_buf_);
    vk_.DestroyBuffer(result_buf_);
    vk_.DestroyBuffer(tet_contrib_grad_buf_);
    vk_.DestroyBuffer(tet_contrib_hess_buf_);
    vk_.DestroyBuffer(reduced_in_buf_);
    vk_.DestroyBuffer(reduced_out_buf_);
    vk_.DestroyBuffer(reduced_q_buf_);
    vk_.DestroyBuffer(reduced_cg_r_buf_);
    vk_.DestroyBuffer(reduced_cg_x_buf_);
    vk_.DestroyBuffer(reduced_cg_ctrl_buf_);
    vk_.DestroyBuffer(reduced_cg_dispatch_buf_);
    vk_.DestroyBuffer(reduced_partial_buf_);
    vk_.DestroyBuffer(basis_buf_);
    vk_.DestroyBuffer(x_rest_buf_);
    vk_.DestroyBuffer(mass_buf_);
    vk_.DestroyBuffer(tets_buf_);
    vk_.DestroyBuffer(dminv_buf_);
    vk_.DestroyBuffer(vol_buf_);
    vk_.DestroyBuffer(csr_offsets_buf_);
    vk_.DestroyBuffer(csr_tets_buf_);
    vk_.DestroyBuffer(csr_local_buf_);
}

bool VkTwoPhaseOps::UpdateDescriptors(std::string& error) {
    auto update = [&](ComputePipeline& pipe, const std::vector<VulkanContext::Buffer*>& buffers) -> bool {
        std::vector<VkDescriptorBufferInfo> infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            infos[i].buffer = buffers[i]->buffer;
            infos[i].offset = 0;
            infos[i].range = buffers[i]->size;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = pipe.descriptor_set;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(vk_.Device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    };

    if (!update(grad_stage_a_, {&x_buf_, &tets_buf_, &dminv_buf_, &vol_buf_, &tet_contrib_grad_buf_})) {
        error = "Failed to update gradient stage A descriptors.";
        return false;
    }
    if (!update(grad_stage_b_, {&x_buf_, &x_star_buf_, &mass_buf_, &csr_offsets_buf_, &csr_tets_buf_, &csr_local_buf_, &tet_contrib_grad_buf_, &force_buf_})) {
        error = "Failed to update gradient stage B descriptors.";
        return false;
    }
    if (!update(hess_stage_a_, {&x_buf_, &p_buf_, &tets_buf_, &dminv_buf_, &vol_buf_, &tet_contrib_hess_buf_})) {
        error = "Failed to update hessian stage A descriptors.";
        return false;
    }
    if (!update(hess_stage_b_, {&x_buf_, &p_buf_, &mass_buf_, &csr_offsets_buf_, &csr_tets_buf_, &csr_local_buf_, &tet_contrib_hess_buf_, &result_buf_})) {
        error = "Failed to update hessian stage B descriptors.";
        return false;
    }
    if (!update(predict_state_, {&x_n_buf_, &v_n_buf_, &x_star_buf_, &x_buf_})) {
        error = "Failed to update predict-state descriptors.";
        return false;
    }
    if (!update(update_velocity_state_, {&x_buf_, &x_n_buf_, &v_n_buf_})) {
        error = "Failed to update update-velocity-state descriptors.";
        return false;
    }
    if (!update(reconstruct_x_, {&basis_buf_, &x_rest_buf_, &reduced_q_buf_, &x_buf_})) {
        error = "Failed to update reconstruct-x descriptors.";
        return false;
    }
    if (!update(build_world_, {&basis_buf_, &reduced_in_buf_, &p_buf_})) {
        error = "Failed to update build-world descriptors.";
        return false;
    }
    if (!update(project_force_stage1_, {&basis_buf_, &force_buf_, &reduced_partial_buf_})) {
        error = "Failed to update project-force stage1 descriptors.";
        return false;
    }
    if (!update(project_result_stage1_, {&basis_buf_, &result_buf_, &reduced_partial_buf_})) {
        error = "Failed to update project-result stage1 descriptors.";
        return false;
    }
    if (!update(project_stage2_, {&reduced_partial_buf_, &reduced_out_buf_})) {
        error = "Failed to update project stage2 descriptors.";
        return false;
    }
    if (!update(reduced_cg_init_,
                {&reduced_out_buf_,
                 &reduced_in_buf_,
                 &reduced_cg_r_buf_,
                 &reduced_cg_x_buf_,
                 &reduced_cg_ctrl_buf_,
                 &reduced_cg_dispatch_buf_})) {
        error = "Failed to update reduced-CG init descriptors.";
        return false;
    }
    if (!update(reduced_cg_update_,
                {&reduced_in_buf_,
                 &reduced_out_buf_,
                 &reduced_cg_r_buf_,
                 &reduced_cg_x_buf_,
                 &reduced_cg_ctrl_buf_,
                 &reduced_cg_dispatch_buf_})) {
        error = "Failed to update reduced-CG update descriptors.";
        return false;
    }
    return true;
}

std::vector<VkTwoPhaseOps::Vec4Std> VkTwoPhaseOps::ToVec4(const std::vector<Vec3>& v) {
    std::vector<Vec4Std> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i].x = v[i].x;
        out[i].y = v[i].y;
        out[i].z = v[i].z;
        out[i].w = 0.0f;
    }
    return out;
}

std::vector<Vec3> VkTwoPhaseOps::FromVec4(const std::vector<Vec4Std>& v) {
    std::vector<Vec3> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i].x = v[i].x;
        out[i].y = v[i].y;
        out[i].z = v[i].z;
    }
    return out;
}

bool VkTwoPhaseOps::Initialize(const MeshData& mesh,
                               const CsrAdjacency& csr,
                               const SubspaceBasisData& basis,
                               const std::vector<Vec3>& x_rest,
                               const SimParams& params,
                               std::string& error) {
    Shutdown();

    if (!vk_.Initialize(error)) {
        return false;
    }

    num_verts_ = mesh.verts.size();
    num_tets_ = mesh.tets.size();
    num_reduced_ = static_cast<std::size_t>(basis.cols);
    num_project_chunks_ = (num_verts_ + 255) / 256;
    if (x_rest.size() != num_verts_) {
        error = "x_rest size mismatch.";
        Shutdown();
        return false;
    }
    if (num_reduced_ == 0) {
        error = "Reduced dimension is zero.";
        Shutdown();
        return false;
    }
    if (basis.basis_row_major.size() != num_verts_ * 3 * num_reduced_) {
        error = "Basis size does not match mesh DOF and reduced dimension.";
        Shutdown();
        return false;
    }
    push_.num_verts = static_cast<std::uint32_t>(num_verts_);
    push_.num_tets = static_cast<std::uint32_t>(num_tets_);
    push_.inv_dt2 = 1.0f / (params.dt * params.dt);
    push_.mu = params.mu;
    push_.lam = params.lam;
    push_.alpha = params.alpha_gaia;
    push_.ground_y = params.ground_y;
    push_.ground_k = params.ground_k;
    push_.use_ground = params.use_ground ? 1u : 0u;
    push_.num_reduced = static_cast<std::uint32_t>(num_reduced_);
    push_.dt = params.dt;
    push_.inv_dt = 1.0f / params.dt;
    push_.gravity_x = params.gravity.x;
    push_.gravity_y = params.gravity.y;
    push_.gravity_z = params.gravity.z;
    push_.cg_damping = params.reduced_damping;
    push_.cg_rel_tol = std::max(params.cg_rel_tol, 0.0f);
    push_.cg_max_iters = static_cast<std::uint32_t>(std::max(params.cg_iters, 0));

    if (!CreateAndUploadStaticBuffers(mesh, csr, basis, x_rest, error)) {
        Shutdown();
        return false;
    }
    if (!CreateDynamicBuffers(num_verts_, num_tets_, num_reduced_, error)) {
        Shutdown();
        return false;
    }

    const std::filesystem::path shader_dir(VKCS_SHADER_DIR);
    if (!CreateComputePipeline((shader_dir / "gradient_tet_stage.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2), StorageBinding(3), StorageBinding(4)},
                               grad_stage_a_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "gradient_vertex_gather.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2), StorageBinding(3), StorageBinding(4), StorageBinding(5), StorageBinding(6), StorageBinding(7)},
                               grad_stage_b_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "hessp_tet_stage.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2), StorageBinding(3), StorageBinding(4), StorageBinding(5)},
                               hess_stage_a_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "hessp_vertex_gather.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2), StorageBinding(3), StorageBinding(4), StorageBinding(5), StorageBinding(6), StorageBinding(7)},
                               hess_stage_b_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "predict_state.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2), StorageBinding(3)},
                               predict_state_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "update_velocity_state.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2)},
                               update_velocity_state_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "reconstruct_x_from_reduced.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2), StorageBinding(3)},
                               reconstruct_x_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "build_world_from_reduced.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2)},
                               build_world_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "project_world_to_reduced_stage1.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2)},
                               project_force_stage1_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "project_world_to_reduced_stage1.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1), StorageBinding(2)},
                               project_result_stage1_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "project_world_to_reduced_stage2.comp.spv").string(),
                               {StorageBinding(0), StorageBinding(1)},
                               project_stage2_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "reduced_cg_init.comp.spv").string(),
                               {StorageBinding(0),
                                StorageBinding(1),
                                StorageBinding(2),
                                StorageBinding(3),
                                StorageBinding(4),
                                StorageBinding(5)},
                               reduced_cg_init_,
                               error)) {
        Shutdown();
        return false;
    }
    if (!CreateComputePipeline((shader_dir / "reduced_cg_update.comp.spv").string(),
                               {StorageBinding(0),
                                StorageBinding(1),
                                StorageBinding(2),
                                StorageBinding(3),
                                StorageBinding(4),
                                StorageBinding(5)},
                               reduced_cg_update_,
                               error)) {
        Shutdown();
        return false;
    }

    if (!UpdateDescriptors(error)) {
        Shutdown();
        return false;
    }

    initialized_ = true;
    return true;
}

void VkTwoPhaseOps::Shutdown() {
    if (vk_.Device() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vk_.Device());
    }

    DestroyComputePipeline(grad_stage_a_);
    DestroyComputePipeline(grad_stage_b_);
    DestroyComputePipeline(hess_stage_a_);
    DestroyComputePipeline(hess_stage_b_);
    DestroyComputePipeline(predict_state_);
    DestroyComputePipeline(update_velocity_state_);
    DestroyComputePipeline(reconstruct_x_);
    DestroyComputePipeline(build_world_);
    DestroyComputePipeline(project_force_stage1_);
    DestroyComputePipeline(project_result_stage1_);
    DestroyComputePipeline(project_stage2_);
    DestroyComputePipeline(reduced_cg_init_);
    DestroyComputePipeline(reduced_cg_update_);
    DestroyBuffers();
    vk_.Shutdown();
    initialized_ = false;
    num_verts_ = 0;
    num_tets_ = 0;
    num_reduced_ = 0;
    num_project_chunks_ = 0;
}

bool VkTwoPhaseOps::UploadState(const std::vector<Vec3>& x, const std::vector<Vec3>& x_star, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (x.size() != num_verts_ || x_star.size() != num_verts_) {
        error = "UploadState size mismatch.";
        return false;
    }
    const std::vector<Vec4Std> x4 = ToVec4(x);
    const std::vector<Vec4Std> x_star4 = ToVec4(x_star);
    std::vector<Vec4Std> v_zero(num_verts_);
    if (!vk_.UploadToBuffer(x_buf_, x4.data(), x_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(x_star_buf_, x_star4.data(), x_star_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(x_n_buf_, x4.data(), x_n_buf_.size, error)) return false;
    if (!vk_.UploadToBuffer(v_n_buf_, v_zero.data(), v_n_buf_.size, error)) return false;
    return true;
}

bool VkTwoPhaseOps::PredictState(std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    return RunPredictDispatch(error);
}

bool VkTwoPhaseOps::UpdateVelocityState(std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    return RunUpdateVelocityDispatch(error);
}

void VkTwoPhaseOps::InsertComputeBarrier(VkCommandBuffer cmd) const {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);
}

void VkTwoPhaseOps::InsertComputeToIndirectBarrier(VkCommandBuffer cmd) const {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);
}

void VkTwoPhaseOps::RecordPredictDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_v =
        static_cast<std::uint32_t>((num_verts_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, predict_state_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            predict_state_.pipeline_layout,
                            0,
                            1,
                            &predict_state_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       predict_state_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_v), 1, 1);
}

void VkTwoPhaseOps::RecordUpdateVelocityDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_v =
        static_cast<std::uint32_t>((num_verts_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, update_velocity_state_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            update_velocity_state_.pipeline_layout,
                            0,
                            1,
                            &update_velocity_state_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       update_velocity_state_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_v), 1, 1);
}

bool VkTwoPhaseOps::RunPredictDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordPredictDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

bool VkTwoPhaseOps::RunUpdateVelocityDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordUpdateVelocityDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

void VkTwoPhaseOps::RecordGradientDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_t =
        static_cast<std::uint32_t>((num_tets_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);
    const std::uint32_t groups_v =
        static_cast<std::uint32_t>((num_verts_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, grad_stage_a_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            grad_stage_a_.pipeline_layout,
                            0,
                            1,
                            &grad_stage_a_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd, grad_stage_a_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_t), 1, 1);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, grad_stage_b_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            grad_stage_b_.pipeline_layout,
                            0,
                            1,
                            &grad_stage_b_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd, grad_stage_b_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_v), 1, 1);
}

void VkTwoPhaseOps::RecordHessianDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_t =
        static_cast<std::uint32_t>((num_tets_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);
    const std::uint32_t groups_v =
        static_cast<std::uint32_t>((num_verts_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hess_stage_a_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            hess_stage_a_.pipeline_layout,
                            0,
                            1,
                            &hess_stage_a_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd, hess_stage_a_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_t), 1, 1);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hess_stage_b_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            hess_stage_b_.pipeline_layout,
                            0,
                            1,
                            &hess_stage_b_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd, hess_stage_b_.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_v), 1, 1);
}

bool VkTwoPhaseOps::RunGradientDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordGradientDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

bool VkTwoPhaseOps::RunHessianDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordHessianDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

void VkTwoPhaseOps::RecordReconstructXDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_v =
        static_cast<std::uint32_t>((num_verts_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reconstruct_x_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            reconstruct_x_.pipeline_layout,
                            0,
                            1,
                            &reconstruct_x_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       reconstruct_x_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_v), 1, 1);
}

bool VkTwoPhaseOps::RunReconstructXDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordReconstructXDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

void VkTwoPhaseOps::RecordGradientReducedDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_r =
        static_cast<std::uint32_t>((num_reduced_ + kProjectWorkgroupSize - 1) / kProjectWorkgroupSize);
    const std::uint32_t groups_chunks = static_cast<std::uint32_t>(std::max<std::size_t>(num_project_chunks_, 1));

    RecordGradientDispatch(cmd);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_force_stage1_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            project_force_stage1_.pipeline_layout,
                            0,
                            1,
                            &project_force_stage1_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       project_force_stage1_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, groups_chunks, static_cast<std::uint32_t>(std::max<std::size_t>(num_reduced_, 1)), 1);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_stage2_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            project_stage2_.pipeline_layout,
                            0,
                            1,
                            &project_stage2_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       project_stage2_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_r), 1, 1);
}

bool VkTwoPhaseOps::RunGradientReducedDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordGradientReducedDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

void VkTwoPhaseOps::RecordHessianReducedDispatch(VkCommandBuffer cmd) const {
    const std::uint32_t groups_v =
        static_cast<std::uint32_t>((num_verts_ + kTetVertWorkgroupSize - 1) / kTetVertWorkgroupSize);
    const std::uint32_t groups_r =
        static_cast<std::uint32_t>((num_reduced_ + kProjectWorkgroupSize - 1) / kProjectWorkgroupSize);
    const std::uint32_t groups_chunks = static_cast<std::uint32_t>(std::max<std::size_t>(num_project_chunks_, 1));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_world_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            build_world_.pipeline_layout,
                            0,
                            1,
                            &build_world_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       build_world_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_v), 1, 1);
    InsertComputeBarrier(cmd);

    RecordHessianDispatch(cmd);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_result_stage1_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            project_result_stage1_.pipeline_layout,
                            0,
                            1,
                            &project_result_stage1_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       project_result_stage1_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, groups_chunks, static_cast<std::uint32_t>(std::max<std::size_t>(num_reduced_, 1)), 1);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_stage2_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            project_stage2_.pipeline_layout,
                            0,
                            1,
                            &project_stage2_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       project_stage2_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, std::max(1u, groups_r), 1, 1);
}

void VkTwoPhaseOps::RecordHessianReducedDispatchIndirect(VkCommandBuffer cmd) const {
    const VkDeviceSize off_build = static_cast<VkDeviceSize>(offsetof(CgDispatchArgsStd, build_world));
    const VkDeviceSize off_hess_tet = static_cast<VkDeviceSize>(offsetof(CgDispatchArgsStd, hess_tet));
    const VkDeviceSize off_hess_vert = static_cast<VkDeviceSize>(offsetof(CgDispatchArgsStd, hess_vert));
    const VkDeviceSize off_proj_stage1 = static_cast<VkDeviceSize>(offsetof(CgDispatchArgsStd, project_stage1));
    const VkDeviceSize off_proj_stage2 = static_cast<VkDeviceSize>(offsetof(CgDispatchArgsStd, project_stage2));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_world_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            build_world_.pipeline_layout,
                            0,
                            1,
                            &build_world_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       build_world_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatchIndirect(cmd, reduced_cg_dispatch_buf_.buffer, off_build);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hess_stage_a_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            hess_stage_a_.pipeline_layout,
                            0,
                            1,
                            &hess_stage_a_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       hess_stage_a_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatchIndirect(cmd, reduced_cg_dispatch_buf_.buffer, off_hess_tet);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hess_stage_b_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            hess_stage_b_.pipeline_layout,
                            0,
                            1,
                            &hess_stage_b_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       hess_stage_b_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatchIndirect(cmd, reduced_cg_dispatch_buf_.buffer, off_hess_vert);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_result_stage1_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            project_result_stage1_.pipeline_layout,
                            0,
                            1,
                            &project_result_stage1_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       project_result_stage1_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatchIndirect(cmd, reduced_cg_dispatch_buf_.buffer, off_proj_stage1);
    InsertComputeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, project_stage2_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            project_stage2_.pipeline_layout,
                            0,
                            1,
                            &project_stage2_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       project_stage2_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatchIndirect(cmd, reduced_cg_dispatch_buf_.buffer, off_proj_stage2);
}

bool VkTwoPhaseOps::RunHessianReducedDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordHessianReducedDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

void VkTwoPhaseOps::RecordReducedCgInitDispatch(VkCommandBuffer cmd) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduced_cg_init_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            reduced_cg_init_.pipeline_layout,
                            0,
                            1,
                            &reduced_cg_init_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       reduced_cg_init_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, 1, 1, 1);
}

void VkTwoPhaseOps::RecordReducedCgUpdateDispatch(VkCommandBuffer cmd) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduced_cg_update_.pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            reduced_cg_update_.pipeline_layout,
                            0,
                            1,
                            &reduced_cg_update_.descriptor_set,
                            0,
                            nullptr);
    vkCmdPushConstants(cmd,
                       reduced_cg_update_.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(PushConstants),
                       &push_);
    vkCmdDispatch(cmd, 1, 1, 1);
}

bool VkTwoPhaseOps::RunReducedCgInitDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordReducedCgInitDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

bool VkTwoPhaseOps::RunReducedCgUpdateDispatch(std::string& error) {
    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordReducedCgUpdateDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

bool VkTwoPhaseOps::ReconstructXFromReduced(const std::vector<float>& q_reduced, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (q_reduced.size() != num_reduced_) {
        error = "ReconstructXFromReduced size mismatch.";
        return false;
    }
    if (!vk_.WriteHostVisibleBuffer(reduced_q_buf_, q_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }
    return RunReconstructXDispatch(error);
}

bool VkTwoPhaseOps::ComputeGradientReducedFromQ(const std::vector<float>& q_reduced,
                                                std::vector<float>& out_reduced,
                                                std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (q_reduced.size() != num_reduced_) {
        error = "ComputeGradientReducedFromQ size mismatch.";
        return false;
    }
    if (!vk_.WriteHostVisibleBuffer(reduced_q_buf_, q_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }

    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordReconstructXDispatch(cmd);
    InsertComputeBarrier(cmd);
    RecordGradientReducedDispatch(cmd);
    if (!vk_.EndSingleTimeCommands(cmd, error)) {
        return false;
    }

    out_reduced.resize(num_reduced_);
    if (!vk_.ReadHostVisibleBuffer(reduced_out_buf_, out_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }
    return true;
}

bool VkTwoPhaseOps::ReconstructAndUpdateVelocityFromQ(const std::vector<float>& q_reduced, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (q_reduced.size() != num_reduced_) {
        error = "ReconstructAndUpdateVelocityFromQ size mismatch.";
        return false;
    }
    if (!vk_.WriteHostVisibleBuffer(reduced_q_buf_, q_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }

    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordReconstructXDispatch(cmd);
    InsertComputeBarrier(cmd);
    RecordUpdateVelocityDispatch(cmd);
    return vk_.EndSingleTimeCommands(cmd, error);
}

bool VkTwoPhaseOps::DownloadX(std::vector<Vec3>& out_x, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    std::vector<Vec4Std> tmp(num_verts_);
    if (!vk_.DownloadFromBuffer(x_buf_, tmp.data(), x_buf_.size, error)) {
        return false;
    }
    out_x = FromVec4(tmp);
    return true;
}

bool VkTwoPhaseOps::ComputeGradient(std::vector<Vec3>& out_force, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (!RunGradientDispatch(error)) {
        return false;
    }
    std::vector<Vec4Std> tmp(num_verts_);
    if (!vk_.DownloadFromBuffer(force_buf_, tmp.data(), force_buf_.size, error)) {
        return false;
    }
    out_force = FromVec4(tmp);
    return true;
}

bool VkTwoPhaseOps::ComputeHessianVec(const std::vector<Vec3>& p, std::vector<Vec3>& out_result, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (p.size() != num_verts_) {
        error = "ComputeHessianVec size mismatch.";
        return false;
    }

    const std::vector<Vec4Std> p4 = ToVec4(p);
    if (!vk_.UploadToBuffer(p_buf_, p4.data(), p_buf_.size, error)) {
        return false;
    }
    if (!RunHessianDispatch(error)) {
        return false;
    }
    std::vector<Vec4Std> tmp(num_verts_);
    if (!vk_.DownloadFromBuffer(result_buf_, tmp.data(), result_buf_.size, error)) {
        return false;
    }
    out_result = FromVec4(tmp);
    return true;
}

bool VkTwoPhaseOps::ComputeGradientReduced(std::vector<float>& out_reduced, std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (!RunGradientReducedDispatch(error)) {
        return false;
    }
    out_reduced.resize(num_reduced_);
    if (!vk_.ReadHostVisibleBuffer(reduced_out_buf_, out_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }
    return true;
}

bool VkTwoPhaseOps::ComputeHessianReduced(const std::vector<float>& p_reduced,
                                          std::vector<float>& out_reduced,
                                          std::string& error) {
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (p_reduced.size() != num_reduced_) {
        error = "ComputeHessianReduced size mismatch.";
        return false;
    }
    if (!vk_.WriteHostVisibleBuffer(reduced_in_buf_, p_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }
    if (!RunHessianReducedDispatch(error)) {
        return false;
    }
    out_reduced.resize(num_reduced_);
    if (!vk_.ReadHostVisibleBuffer(reduced_out_buf_, out_reduced.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }
    return true;
}

bool VkTwoPhaseOps::SolveReducedCgGpuFixed(const std::vector<float>& rhs,
                                           int cg_iters,
                                           std::vector<float>& out_x,
                                           CgSolveStats& out_stats,
                                           std::string& error) {
    out_stats = CgSolveStats{};
    if (!initialized_) {
        error = "VkTwoPhaseOps is not initialized.";
        return false;
    }
    if (rhs.size() != num_reduced_) {
        error = "SolveReducedCgGpuFixed rhs size mismatch.";
        return false;
    }

    const int iters = std::max(cg_iters, 0);
    if (iters == 0) {
        out_x.assign(num_reduced_, 0.0f);
        return true;
    }

    if (!vk_.WriteHostVisibleBuffer(reduced_out_buf_, rhs.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }

    VkCommandBuffer cmd = vk_.BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    RecordReducedCgInitDispatch(cmd);
    InsertComputeToIndirectBarrier(cmd);
    for (int i = 0; i < iters; ++i) {
        RecordHessianReducedDispatchIndirect(cmd);
        InsertComputeBarrier(cmd);
        RecordReducedCgUpdateDispatch(cmd);
        InsertComputeToIndirectBarrier(cmd);
    }
    if (!vk_.EndSingleTimeCommands(cmd, error)) {
        return false;
    }

    out_x.resize(num_reduced_);
    if (!vk_.ReadHostVisibleBuffer(reduced_cg_x_buf_, out_x.data(), num_reduced_ * sizeof(float), error)) {
        return false;
    }

    CgControlStd ctrl{};
    if (!vk_.ReadHostVisibleBuffer(reduced_cg_ctrl_buf_, &ctrl, sizeof(ctrl), error)) {
        return false;
    }
    out_stats.iterations = std::min<std::uint32_t>(ctrl.iter, static_cast<std::uint32_t>(iters));
    out_stats.hessian_calls = out_stats.iterations;
    return true;
}
