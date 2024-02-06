#pragma once

#include "glfw/glfw3.h"

class MyWindow
{
private:
    /* data */
    GLFWwindow *window;
    unsigned int width;
    unsigned int height;

public:
    MyWindow(unsigned int width, unsigned int height, const char *title);
    ~MyWindow();

    GLFWwindow *GetWindow() { return window; }

    unsigned int GetWidth() const { return width; }

    unsigned int GetHeight() const { return height; }
};
