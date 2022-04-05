
#include <iostream>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image_loader.h>
#include "stb_image_resize.h"
#include "ImageDatabase.h"
#include <chrono>
#include <execution>
#include <future>

template <typename TP>
auto lapTime(TP &tp)
{
    TP new_tp = TP::clock::now();
    auto d = new_tp - tp;
    tp = new_tp;
    return d;
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cout << "Usage : " << argv[0] << "db imgIn inBlockSize outBlockSize imgOut" << std::endl;
        return -1;
    }
    ImageData img = loadImage(argv[2]);
    int block_size = atoi(argv[3]);
    int mosaic_block_size = atoi(argv[4]);
    if (img.width % block_size || img.height % block_size)
    {
        std::cout << "Use an image whose size is a multiple of the block size!" << std::endl;
        return -1;
    }

    auto tp = std::chrono::high_resolution_clock::now();
    ImageDatabase db(argv[1]);
    std::cout << "Loading db took " << std::chrono::duration<float>(lapTime(tp)).count() << "s" << std::endl;

    std::vector<ImageBlockView> blocks(img.height / block_size * img.width / block_size);
    for (int j = 0; j < img.height / block_size; ++j)
    {
        for (int i = 0; i < img.width / block_size; ++i)
        {
            ImageBlockView& block = blocks[j * img.width / block_size + i];
            block.img = &img;
            block.height = block.width = block_size;
            block.x_start = i * block_size;
            block.y_start = j * block_size;
        }
    }
    std::vector<std::size_t> indices(blocks.size());
    std::transform(std::execution::par_unseq, begin(blocks), end(blocks), begin(indices), [&](const auto& block) { return db.findBestEntryUnique(block); });

    std::cout << "Finding the best blocks took " << std::chrono::duration<float>(lapTime(tp)).count() << "s" << std::endl;

    OrderedDirectory dir(argv[1]);
    ImageData mosaic;
    mosaic.height = img.height / block_size * mosaic_block_size;
    mosaic.width = img.width / block_size * mosaic_block_size;
    mosaic.pixels.resize((std::size_t)mosaic.height * mosaic.width);
    std::vector<std::future<void>> tasks;
    tasks.reserve(img.height / block_size * img.width / block_size);
    for (int j = 0; j < img.height / block_size; ++j)
    {
        for (int i = 0; i < img.width / block_size; ++i)
        {
            tasks.emplace_back(std::async(std::launch::async, [&,i,j](){
                ImageBlockView block;
                block.img = &mosaic;
                block.height = block.width = mosaic_block_size;
                block.x_start = i * mosaic_block_size;
                block.y_start = j * mosaic_block_size;
                ImageData tile = loadImage(dir[indices[j * img.width / block_size + i]].string().c_str());
                stbir_resize_float(&tile.pixels[0].r, tile.width, tile.height, 0, &block(0, 0).r, block.width, block.height, block.img->width * sizeof(RgbPixel), 3);
            }));
        }
    }
    tasks.clear(); //Calls async future destructors, waiting for end of tasks
    std::cout << "Building the mosaic took " << std::chrono::duration<float>(lapTime(tp)).count() << "s" << std::endl;

    saveImage(mosaic, argv[5]);

    return EXIT_SUCCESS;
}