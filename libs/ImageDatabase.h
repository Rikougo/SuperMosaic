#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H

#include <cstdint>
#include <filesystem>
#include <vector>
#include "image_loader.h"

struct ImageBlockView
{
	const ImageData* img;
	int x_start, y_start;
	int width, height;
    RgbPixel operator()(int i, int j);
};

constexpr std::size_t ThumbnailSize = 16;
struct ImageEntry
{
    RgbPixel thumbnail[ThumbnailSize * ThumbnailSize];
    RgbPixel mean;
    RgbPixel std_deviation;
    const RgbPixel& operator()(std::size_t i, std::size_t j) const;
    RgbPixel& operator()(std::size_t i, std::size_t j);
};

class OrderedDirectory
{
public:
    OrderedDirectory(const std::filesystem::path& directory);
    auto begin() const { return std::begin(_files); }
    auto end() const { return std::end(_files); }
    std::size_t size() const;
    const std::filesystem::path& operator[](std::size_t i);
private:
    std::vector<std::filesystem::path> _files;
};

class ImageDatabase
{
public:
    ImageDatabase(const std::filesystem::path& db_folder);
private:
    std::vector<ImageEntry> _entries;
};



#endif
