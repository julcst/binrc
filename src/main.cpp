#include "mainapp.hpp"

#include <iostream>

int main() {
    try {
        MainApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}