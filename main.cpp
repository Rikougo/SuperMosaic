
#include <iostream>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image_loader.h>
#include "stb_image_resize.h"

int main()
{
    ImageData img = loadImage("./test.png");

    std::cout << "Image " << img.width << "x" << img.height << ", mean : " << img.mean << ", standard deviation : " << img.std_deviation << std::endl;

    saveImage(img, "data.png");

    return EXIT_SUCCESS;
}