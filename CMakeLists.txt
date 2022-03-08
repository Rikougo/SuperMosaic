cmake_minimum_required(VERSION 3.1)

project(SuperMosaic DESCRIPTION "Super mosaic" LANGUAGES CXX)

add_executable(main main.cpp ./libs/image_loader.cpp)

target_include_directories(main PRIVATE libs)