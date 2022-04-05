cmake_minimum_required(VERSION 3.1)

project(SuperMosaic DESCRIPTION "Super mosaic" LANGUAGES CXX)

add_executable(main main.cpp ./libs/image_loader.cpp ./libs/image_loader.h ./libs/ImageDatabase.cpp ./libs/ImageDatabase.h)
find_package( Threads )
target_link_libraries( main ${CMAKE_THREAD_LIBS_INIT} )

set_target_properties(main
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_include_directories(main PRIVATE libs)