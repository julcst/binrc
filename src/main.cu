#include "mainapp.cuh"

#include <iostream>

#include "test.cuh"

int main() {
    try {
        test();
        MainApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}