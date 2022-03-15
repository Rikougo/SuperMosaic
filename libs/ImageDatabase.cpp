#include "ImageDatabase.h"
#include "stb_image_resize.h"
#include <fstream>
#include <algorithm>

namespace
{
    template<typename T>
    std::vector<T> rawVectorLoad(std::ifstream& input)
    {
        std::size_t n;
        input.read(reinterpret_cast<char*>(&n), sizeof(n));
        std::vector<T> result(n);
        input.read(reinterpret_cast<char*>(result.data()), sizeof(T) * n);
        return result;
    }

    template<typename T>
    void rawVectorStore(std::ofstream& output, const std::vector<T>& data)
    {
        std::size_t n = data.size();
        output.write(reinterpret_cast<const char*>(&n), sizeof(n));
        output.write(reinterpret_cast<const char*>(data.data()), sizeof(T) * n);
    }

    ImageEntry computeEntry(const std::filesystem::path& file)
    {
        ImageData img = loadImage(file.string().c_str());
        ImageEntry entry;
        entry.mean = img.mean;
        entry.std_deviation = img.std_deviation;
        stbir_resize_float(&img.pixels[0].r, img.width, img.height, 0, &entry.thumbnail[0].r, ThumbnailSize, ThumbnailSize, 0, 3);
        return entry;
    }

    std::vector<ImageEntry> loadEntries(const std::filesystem::path& db_folder)
    {
        auto index_path = db_folder.lexically_normal();
        if(!index_path.has_filename()) index_path = index_path.parent_path();
        index_path+=".imgdb";
        if(std::filesystem::is_regular_file(index_path))
        {
            std::ifstream index(index_path, std::ios_base::binary);
            return rawVectorLoad<ImageEntry>(index);
        }
        //No premade index, build it
        OrderedDirectory dir(db_folder);
        std::vector<ImageEntry> result(dir.size());
        auto entry_it = result.begin();
        for(auto& path : dir)
        {
            *(entry_it++) = computeEntry(path);
        }
        std::ofstream index(index_path, std::ios_base::binary);
        rawVectorStore(index, result);

        return result;
    }
}

RgbPixel ImageBlockView::operator()(int i, int j)
{
	return img->pixels[x_start + i + (y_start + j) * img->width];
}

OrderedDirectory::OrderedDirectory(const std::filesystem::path& directory)
    :_files(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{})
{
    std::ranges::sort(_files);
}

std::size_t OrderedDirectory::size() const
{
    return _files.size();
}

const std::filesystem::path& OrderedDirectory::operator[](std::size_t i)
{
    return _files[i];
}

const RgbPixel& ImageEntry::operator()(std::size_t i, std::size_t j) const
{
    return thumbnail[j * ThumbnailSize + i];
}

RgbPixel& ImageEntry::operator()(std::size_t i, std::size_t j)
{
    return const_cast<RgbPixel&>(std::as_const(*this)(i,j));
}

ImageDatabase::ImageDatabase(const std::filesystem::path& db_folder)
    :_entries(loadEntries(db_folder))
{
}
