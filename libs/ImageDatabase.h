#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H

#include <cstdint>
#include <filesystem>
#include <vector>
#include <atomic>
#include "image_loader.h"

struct ImageBlockView
{
	ImageData* img;
	int x_start, y_start;
	int width, height;
    RgbPixel& operator()(int i, int j);
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

    std::vector<std::size_t> findAllEntries(const ImageData& img, int block_size);
    std::size_t findBestEntry(ImageBlockView block) const;
    std::size_t findBestEntryUnique(ImageBlockView block);
    std::size_t size() const;

private:
	std::size_t doFindBestEntry(const ImageEntry& img) const;
    std::vector<ImageEntry> _entries;
	std::unique_ptr<std::atomic_bool[]> _used;
};


#endif
