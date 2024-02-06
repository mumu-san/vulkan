
#include "MyWindow.h"

MyWindow::MyWindow(unsigned int width, unsigned int height, const char *title)
    : width(width), height(height)
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
}

MyWindow::~MyWindow()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}