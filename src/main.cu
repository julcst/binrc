#include "mainapp.cuh"

#include <iostream>
#include "brdflut.cuh"

int main() {
    try {
        computeAndSaveDirectionalAlbedo("lut.png", 1024, 1024);
        MainApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}