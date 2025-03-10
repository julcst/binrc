#include "mainapp.cuh"

#include <iostream>

int main() {
    try {
        MainApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}