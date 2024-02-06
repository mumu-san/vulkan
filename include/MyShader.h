#pragma once

#include <string_view>
#include <vector>

class MyShader
{
private:
    std::vector<std::byte> shader_data;

public:
    MyShader(std::string_view shaderPath);

    const std::vector<std::byte> &GetData() const { return shader_data; }
};
