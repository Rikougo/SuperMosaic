
#include <iostream>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image_loader.h>
#include "stb_image_resize.h"
#include "ImageDatabase.h"
#include <chrono>

template<typename TP>
auto lapTime(TP& tp)
{
    TP new_tp = TP::clock::now();
    auto d = new_tp - tp;
    tp = new_tp;
    return d;
}

int main()
{
    ImageData img = loadImage("../images/renard_noel.png");
    std::cout << "Image " << img.width << "x" << img.height << ", mean : " << img.mean << ", standard deviation : " << img.std_deviation << std::endl;

    auto tp = std::chrono::high_resolution_clock::now();
    ImageDatabase db("../images/pets");
    std::cout << "Loading db took " << std::chrono::duration<float>(lapTime(tp)).count() << "s" << std::endl;


    std::vector<std::size_t> indices(img.height / 16 * img.width / 16);
    for (int j = 0; j < img.height / 16; ++j)
    {
        for (int i = 0; i < img.width / 16; ++i)
        {
            ImageBlockView block;
            block.img = &img;
            block.height = block.width = 16;
            block.x_start = i * 16;
            block.y_start = j * 16;
            indices[j * img.width / 16 + i] = db.findBestEntry(block);
        }
    }
    std::cout << "Finding the best blocks took " << std::chrono::duration<float>(lapTime(tp)).count() << "s" << std::endl;

    OrderedDirectory dir("../images/pets");
    ImageData mosaic;
    mosaic.pixels.resize(img.height * 8ull * img.width * 8);
    mosaic.height = img.height * 8;
    mosaic.width = img.width * 8;
    for (int j = 0; j < img.height / 16; ++j)
    {
        for (int i = 0; i < img.width / 16; ++i)
        {
            ImageBlockView block;
            block.img = &mosaic;
            block.height = block.width = 128;
            block.x_start = i * 128;
            block.y_start = j * 128;
            ImageData tile = loadImage(dir[indices[j * img.width / 16 + i]].string().c_str());
            stbir_resize_float(&tile.pixels[0].r, tile.width, tile.height, 0, &block(0,0).r, block.width, block.height, block.img->width * sizeof(RgbPixel), 3);
        }
    }
    std::cout << "Building the mosaic took " << std::chrono::duration<float>(lapTime(tp)).count() << "s" << std::endl;

    saveImage(mosaic, "data.png");

    return EXIT_SUCCESS;
}