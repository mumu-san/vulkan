#include "MyShader.h"

#include <fstream>

MyShader::MyShader(std::string_view shaderPath)
{
    std::ifstream shaderFile(shaderPath.data(), std::ios::ate | std::ios::binary);

    if (!shaderFile.is_open())
    {
        throw std::runtime_error("Failed to open shader file!");
    }

    size_t fileSize = (size_t)shaderFile.tellg();
    shader_data.resize(fileSize);

    shaderFile.seekg(0);
    shaderFile.read((char *)shader_data.data(), fileSize);

    shaderFile.close();
}
