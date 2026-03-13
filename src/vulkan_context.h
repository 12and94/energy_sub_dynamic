#pragma once

#include <cstdint>
#include <string>

#include <vulkan/vulkan.h>

class VulkanContext {
public:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
    };

    bool Initialize(std::string& error);
    void Shutdown();

    bool CreateBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memory_props,
                      Buffer& out,
                      std::string& error) const;

    void DestroyBuffer(Buffer& buffer) const;

    bool UploadToBuffer(const Buffer& dst, const void* src_data, VkDeviceSize size, std::string& error) const;
    bool DownloadFromBuffer(const Buffer& src, void* dst_data, VkDeviceSize size, std::string& error) const;
    bool CopyBuffer(const Buffer& src, const Buffer& dst, VkDeviceSize size, std::string& error) const;
    bool WriteHostVisibleBuffer(const Buffer& dst, const void* src_data, VkDeviceSize size, std::string& error) const;
    bool ReadHostVisibleBuffer(const Buffer& src, void* dst_data, VkDeviceSize size, std::string& error) const;

    bool CreateShaderModuleFromFile(const std::string& path, VkShaderModule& out_module, std::string& error) const;

    VkCommandBuffer BeginSingleTimeCommands(std::string& error) const;
    bool EndSingleTimeCommands(VkCommandBuffer cmd, std::string& error) const;

    VkDevice Device() const { return device_; }
    VkPhysicalDevice PhysicalDevice() const { return physical_device_; }
    VkQueue ComputeQueue() const { return compute_queue_; }
    std::uint32_t ComputeQueueFamily() const { return compute_queue_family_; }
    const std::string& PhysicalDeviceName() const { return physical_device_name_; }

private:
    std::uint32_t FindMemoryType(std::uint32_t type_filter, VkMemoryPropertyFlags properties, std::string& error) const;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    std::uint32_t compute_queue_family_ = 0;
    std::string physical_device_name_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer reusable_cmd_ = VK_NULL_HANDLE;
    VkFence reusable_fence_ = VK_NULL_HANDLE;
};
