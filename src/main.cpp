#include "vulkan/vulkan.h"
#include "glm/glm.hpp"
#include "glfw/glfw3.h"

#include <iostream>
#include <array>
#include <vector>
#include <chrono>

#include "MyWindow.h"
#include "MyShader.h"

VkInstance createInstance(std::vector<const char *> &layers, std::vector<const char *> &extensions)
{
    VkInstance instance;
    VkInstanceCreateInfo instanceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    auto ret = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    assert(ret == VK_SUCCESS);
    return instance;
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance)
{
    VkPhysicalDevice physicalDevice;
    uint32_t physicalDeviceCount = 1;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, &physicalDevice);
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    std::cout << "Physical Device Name: " << physicalDeviceProperties.deviceName << std::endl;
    std::cout << "API Version: " << VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion) << "." << VK_VERSION_MINOR(physicalDeviceProperties.apiVersion) << "." << VK_VERSION_PATCH(physicalDeviceProperties.apiVersion) << std::endl;
    return physicalDevice;
}

VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow *window)
{
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface!");
    }
    return surface;
}

VkDevice createDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, std::vector<const char *> &extensions)
{
    VkDevice device;

    uint32_t deviceExtensionCount;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtensionCount, deviceExtensions.data());
    // erase extensions that are not supported
    for (auto it = extensions.begin(); it != extensions.end();)
    {
        if (std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [&](VkExtensionProperties &p)
                         { return strcmp(p.extensionName, *it) == 0; }) == deviceExtensions.end())
        {
            std::cout << "Extension " << *it << " is not supported" << std::endl;
            it = extensions.erase(it);
        }
        else
        {
            it++;
        }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    VkDeviceCreateInfo deviceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    auto ret = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    assert(ret == VK_SUCCESS);
    return device;
}

VkSwapchainKHR createSwapchain(VkDevice device, VkSurfaceKHR surface, VkSurfaceFormatKHR surfaceFormat, VkExtent2D extent)
{
    VkSwapchainKHR swapchain;
    VkSwapchainCreateInfoKHR swapchainCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = 2,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
    };
    auto ret = vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain);
    assert(ret == VK_SUCCESS);
    return swapchain;
}

VkRenderPass createRenderPass(VkDevice device, VkFormat format)
{
    VkAttachmentDescription colorAttachmentDescription{
        .format = format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference colorAttachmentReference{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpassDescription{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentReference,
    };

    VkSubpassDependency subpassDependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPass renderPass;
    VkRenderPassCreateInfo renderPassCreateInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorAttachmentDescription,
        .subpassCount = 1,
        .pSubpasses = &subpassDescription,
        .dependencyCount = 1,
        .pDependencies = &subpassDependency,
    };
    auto ret = vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &renderPass);
    assert(ret == VK_SUCCESS);
    return renderPass;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format)
{
    VkImageView imageView;
    VkImageViewCreateInfo imageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    auto ret = vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView);
    assert(ret == VK_SUCCESS);
    return imageView;
}

VkFramebuffer createFramebuffer(VkDevice device, VkRenderPass renderPass, VkImageView imageView, VkExtent2D extent)
{
    VkFramebuffer framebuffer;
    VkFramebufferCreateInfo framebufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = renderPass,
        .attachmentCount = 1,
        .pAttachments = &imageView,
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    };
    auto ret = vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &framebuffer);
    assert(ret == VK_SUCCESS);
    return framebuffer;
}

std::vector<VkFramebuffer> createFramebuffers(VkDevice device, VkRenderPass renderPass, VkSwapchainKHR swapchain, VkExtent2D extent)
{
    uint32_t imageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    std::vector<VkImage> images(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data());

    std::vector<VkImageView> scImageViews(imageCount);
    for (uint32_t i = 0; i < imageCount; i++)
    {
        scImageViews[i] = createImageView(device, images[i], VK_FORMAT_B8G8R8A8_UNORM);
    }

    std::vector<VkFramebuffer> framebuffers(imageCount);
    for (uint32_t i = 0; i < imageCount; i++)
    {
        framebuffers[i] = createFramebuffer(device, renderPass, scImageViews[i], extent);
    }
    return framebuffers;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<std::byte> &code)
{
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data()),
    };
    auto ret = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);
    assert(ret == VK_SUCCESS);
    return shaderModule;
}

VkPipeline createPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vert_shaderModule, VkShaderModule frag_shaderModule)
{
    VkPipeline pipeline;
    VkPipelineShaderStageCreateInfo vert_shaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shaderModule,
        .pName = "main",
    };

    VkPipelineShaderStageCreateInfo frag_shaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shaderModule,
        .pName = "main",
    };

    VkPipelineShaderStageCreateInfo shaderStages[] = {vert_shaderStageCreateInfo, frag_shaderStageCreateInfo};

    VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0,
    };

    VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo pipelineViewportStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachmentState,
    };

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &pipelineVertexInputStateCreateInfo,
        .pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo,
        .pViewportState = &pipelineViewportStateCreateInfo,
        .pRasterizationState = &pipelineRasterizationStateCreateInfo,
        .pMultisampleState = &pipelineMultisampleStateCreateInfo,
        .pDepthStencilState = &pipelineDepthStencilStateCreateInfo,
        .pColorBlendState = &pipelineColorBlendStateCreateInfo,
        .pDynamicState = &pipelineDynamicStateCreateInfo,
        .layout = pipelineLayout,
        .renderPass = renderPass,
    };
    vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineCreateInfo, nullptr, &pipeline);
    return pipeline;
}

VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex)
{
    VkCommandPool commandPool;
    VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilyIndex,
    };
    auto ret = vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool);
    assert(ret == VK_SUCCESS);
    return commandPool;
}

VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool)
{
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    auto ret = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
    assert(ret == VK_SUCCESS);
    return commandBuffer;
}

VkFence createFence(VkDevice device)
{
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    auto ret = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
    assert(ret == VK_SUCCESS);
    return fence;
}

VkSemaphore createSemaphore(VkDevice device)
{
    VkSemaphore semaphore;
    VkSemaphoreCreateInfo semaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    auto ret = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore);
    assert(ret == VK_SUCCESS);
    return semaphore;
}

VkQueue createQueue(VkDevice device, uint32_t queueFamilyIndex)
{
    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    return queue;
}

int main()
{
    VkExtent2D extent = {1440, 900};

    MyWindow window(extent.width, extent.height, "Vulkan");
    // validation layers
    const char *validationLayers[] = {
        "VK_LAYER_KHRONOS_validation",
    };
    std::vector<const char *> layers(validationLayers, validationLayers + sizeof(validationLayers) / sizeof(validationLayers[0]));
    // enable ext for glfw
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VkInstance instance = createInstance(layers, extensions);
    VkPhysicalDevice physicalDevice = pickPhysicalDevice(instance);

    VkSurfaceKHR surface = createSurface(instance, window.GetWindow());

    std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    VkDevice device = createDevice(physicalDevice, surface, deviceExtensions);

    VkSwapchainKHR swapchain = createSwapchain(device, surface, {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}, extent);

    VkRenderPass renderPass = createRenderPass(device, VK_FORMAT_B8G8R8A8_UNORM);
    std::vector<VkFramebuffer> framebuffers = createFramebuffers(device, renderPass, swapchain, extent);

    MyShader vert_shader("shader/triangle.vert.spv");
    MyShader frag_shader("shader/triangle.frag.spv");
    VkShaderModule vert_shaderModule = createShaderModule(device, vert_shader.GetData());
    VkShaderModule frag_shaderModule = createShaderModule(device, frag_shader.GetData());

    VkPipeline pipeline = createPipeline(device, renderPass, vert_shaderModule, frag_shaderModule);

    VkCommandPool commandPool = createCommandPool(device, 0);
    VkCommandBuffer commandBuffer = createCommandBuffer(device, commandPool);

    VkFence fence = createFence(device);
    VkSemaphore imageAvailableSemaphore = createSemaphore(device);
    VkSemaphore renderFinishedSemaphore = createSemaphore(device);

    VkQueue queue = createQueue(device, 0);

    // ================== main loop ==================

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = renderPass,
        .framebuffer = framebuffers[0],
        .renderArea = {
            .offset = {0, 0},
            .extent = extent,
        },
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data(),
    };

    VkCommandBufferBeginInfo commandBufferBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imageAvailableSemaphore,
        .pWaitDstStageMask = &waitStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &renderFinishedSemaphore,
    };

    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderFinishedSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = 0,
    };

    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(window.GetWidth()),
        .height = static_cast<float>(window.GetHeight()),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = {window.GetWidth(), window.GetHeight()},
    };

    uint32_t frameIndex = 0;

    auto window_p = window.GetWindow();
    while (!glfwWindowShouldClose(window_p))
    {
        frameIndex++;

        auto current_time = std::chrono::system_clock::now();

        glfwPollEvents();

        if (glfwGetKey(window_p, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window_p, GLFW_TRUE);

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &fence);

        uint32_t currentFrameIndex;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &currentFrameIndex);

        vkResetCommandBuffer(commandBuffer, 0);

        vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);

        renderPassBeginInfo.framebuffer = framebuffers[currentFrameIndex];

        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        vkEndCommandBuffer(commandBuffer);

        vkQueueSubmit(queue, 1, &submitInfo, fence);

        presentInfo.pImageIndices = &currentFrameIndex;

        vkQueuePresentKHR(queue, &presentInfo);

        auto end_time = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - current_time).count();
        std::cout << "frame " << frameIndex << " took " << elapsed * 0.001 << "ms" << std::endl;
    }
}