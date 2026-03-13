#include "vulkan_context.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

bool VulkanContext::Initialize(std::string& error) {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "vk_cs";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "vk_cs";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
        error = "vkCreateInstance failed.";
        return false;
    }

    std::uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        error = "No Vulkan physical device found.";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    for (VkPhysicalDevice dev : devices) {
        std::uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_props(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, queue_props.data());
        for (std::uint32_t i = 0; i < queue_count; ++i) {
            if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical_device_ = dev;
                compute_queue_family_ = i;
                break;
            }
        }
        if (physical_device_ != VK_NULL_HANDLE) {
            break;
        }
    }

    if (physical_device_ == VK_NULL_HANDLE) {
        error = "No queue family with VK_QUEUE_COMPUTE_BIT found.";
        return false;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    physical_device_name_ = props.deviceName;

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = compute_queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures features{};
    features.shaderFloat64 = VK_FALSE;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &features;

    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS) {
        error = "vkCreateDevice failed.";
        return false;
    }
    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = compute_queue_family_;
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        error = "vkCreateCommandPool failed.";
        return false;
    }

    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = command_pool_;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &cmd_alloc, &reusable_cmd_) != VK_SUCCESS) {
        error = "vkAllocateCommandBuffers (reusable) failed.";
        return false;
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device_, &fence_info, nullptr, &reusable_fence_) != VK_SUCCESS) {
        error = "vkCreateFence (reusable) failed.";
        return false;
    }

    return true;
}

void VulkanContext::Shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
    if (reusable_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, reusable_fence_, nullptr);
        reusable_fence_ = VK_NULL_HANDLE;
    }
    if (reusable_cmd_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &reusable_cmd_);
        reusable_cmd_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    physical_device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    physical_device_name_.clear();
}

std::uint32_t VulkanContext::FindMemoryType(std::uint32_t type_filter,
                                            VkMemoryPropertyFlags properties,
                                            std::string& error) const {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        const bool type_ok = (type_filter & (1u << i)) != 0;
        const bool prop_ok = (mem_props.memoryTypes[i].propertyFlags & properties) == properties;
        if (type_ok && prop_ok) {
            return i;
        }
    }
    error = "Failed to find compatible Vulkan memory type.";
    return UINT32_MAX;
}

bool VulkanContext::CreateBuffer(VkDeviceSize size,
                                 VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags memory_props,
                                 Buffer& out,
                                 std::string& error) const {
    out = Buffer{};

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &out.buffer) != VK_SUCCESS) {
        error = "vkCreateBuffer failed.";
        return false;
    }

    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(device_, out.buffer, &mem_req);
    const std::uint32_t memory_type = FindMemoryType(mem_req.memoryTypeBits, memory_props, error);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(device_, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(device_, &alloc_info, nullptr, &out.memory) != VK_SUCCESS) {
        error = "vkAllocateMemory failed.";
        vkDestroyBuffer(device_, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(device_, out.buffer, out.memory, 0) != VK_SUCCESS) {
        error = "vkBindBufferMemory failed.";
        vkFreeMemory(device_, out.memory, nullptr);
        vkDestroyBuffer(device_, out.buffer, nullptr);
        out.memory = VK_NULL_HANDLE;
        out.buffer = VK_NULL_HANDLE;
        return false;
    }

    out.size = size;
    return true;
}

void VulkanContext::DestroyBuffer(Buffer& buffer) const {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

VkCommandBuffer VulkanContext::BeginSingleTimeCommands(std::string& error) const {
    if (reusable_cmd_ == VK_NULL_HANDLE) {
        error = "Reusable command buffer is not initialized.";
        return VK_NULL_HANDLE;
    }
    VkCommandBuffer cmd = reusable_cmd_;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        error = "vkBeginCommandBuffer failed.";
        return VK_NULL_HANDLE;
    }

    return cmd;
}

bool VulkanContext::EndSingleTimeCommands(VkCommandBuffer cmd, std::string& error) const {
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        error = "vkEndCommandBuffer failed.";
        return false;
    }

    if (reusable_fence_ == VK_NULL_HANDLE) {
        error = "Reusable fence is not initialized.";
        return false;
    }
    vkResetFences(device_, 1, &reusable_fence_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    if (vkQueueSubmit(compute_queue_, 1, &submit_info, reusable_fence_) != VK_SUCCESS) {
        error = "vkQueueSubmit failed.";
        return false;
    }

    vkWaitForFences(device_, 1, &reusable_fence_, VK_TRUE, UINT64_MAX);
    return true;
}

bool VulkanContext::CopyBuffer(const Buffer& src, const Buffer& dst, VkDeviceSize size, std::string& error) const {
    VkCommandBuffer cmd = BeginSingleTimeCommands(error);
    if (cmd == VK_NULL_HANDLE) {
        return false;
    }
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = size;
    vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region);
    return EndSingleTimeCommands(cmd, error);
}

bool VulkanContext::UploadToBuffer(const Buffer& dst, const void* src_data, VkDeviceSize size, std::string& error) const {
    Buffer staging;
    if (!CreateBuffer(size,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      staging,
                      error)) {
        return false;
    }

    void* mapped = nullptr;
    if (vkMapMemory(device_, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        error = "vkMapMemory (upload) failed.";
        DestroyBuffer(staging);
        return false;
    }
    std::memcpy(mapped, src_data, static_cast<size_t>(size));
    vkUnmapMemory(device_, staging.memory);

    const bool ok = CopyBuffer(staging, dst, size, error);
    DestroyBuffer(staging);
    return ok;
}

bool VulkanContext::DownloadFromBuffer(const Buffer& src, void* dst_data, VkDeviceSize size, std::string& error) const {
    Buffer staging;
    if (!CreateBuffer(size,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      staging,
                      error)) {
        return false;
    }

    const bool copied = CopyBuffer(src, staging, size, error);
    if (!copied) {
        DestroyBuffer(staging);
        return false;
    }

    void* mapped = nullptr;
    if (vkMapMemory(device_, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        error = "vkMapMemory (download) failed.";
        DestroyBuffer(staging);
        return false;
    }
    std::memcpy(dst_data, mapped, static_cast<size_t>(size));
    vkUnmapMemory(device_, staging.memory);

    DestroyBuffer(staging);
    return true;
}

bool VulkanContext::WriteHostVisibleBuffer(const Buffer& dst,
                                           const void* src_data,
                                           VkDeviceSize size,
                                           std::string& error) const {
    if (size > dst.size) {
        error = "WriteHostVisibleBuffer size exceeds destination buffer.";
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(device_, dst.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        error = "vkMapMemory (host write) failed.";
        return false;
    }
    std::memcpy(mapped, src_data, static_cast<size_t>(size));
    vkUnmapMemory(device_, dst.memory);
    return true;
}

bool VulkanContext::ReadHostVisibleBuffer(const Buffer& src,
                                          void* dst_data,
                                          VkDeviceSize size,
                                          std::string& error) const {
    if (size > src.size) {
        error = "ReadHostVisibleBuffer size exceeds source buffer.";
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(device_, src.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        error = "vkMapMemory (host read) failed.";
        return false;
    }
    std::memcpy(dst_data, mapped, static_cast<size_t>(size));
    vkUnmapMemory(device_, src.memory);
    return true;
}

bool VulkanContext::CreateShaderModuleFromFile(const std::string& path,
                                               VkShaderModule& out_module,
                                               std::string& error) const {
    out_module = VK_NULL_HANDLE;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        error = "Failed to open shader file: " + path;
        return false;
    }
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<char> code(size);
    in.read(code.data(), static_cast<std::streamsize>(size));
    if (!in) {
        error = "Failed to read shader file: " + path;
        return false;
    }

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
    if (vkCreateShaderModule(device_, &info, nullptr, &out_module) != VK_SUCCESS) {
        error = "vkCreateShaderModule failed for: " + path;
        return false;
    }
    return true;
}
