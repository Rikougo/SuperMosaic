#include "ImageDatabase.h"
#include "stb_image_resize.h"
#include <fstream>
#include <algorithm>
#include <execution>

namespace
{
    template <typename T>
    std::vector<T> rawVectorLoad(std::ifstream &input)
    {
        std::size_t n;
        input.read(reinterpret_cast<char *>(&n), sizeof(n));
        std::vector<T> result(n);
        input.read(reinterpret_cast<char *>(result.data()), sizeof(T) * n);
        return result;
    }

    template <typename T>
    void rawVectorStore(std::ofstream &output, const std::vector<T> &data)
    {
        std::size_t n = data.size();
        output.write(reinterpret_cast<const char *>(&n), sizeof(n));
        output.write(reinterpret_cast<const char *>(data.data()), sizeof(T) * n);
    }

    float computePSNR(const ImageEntry &l, const ImageEntry &r)
    {
        using std::abs;
        float sum = 0.0f;

        for (size_t i = 0; i < ThumbnailSize * ThumbnailSize; i++)
        {
            auto diff = l.thumbnail[i] - r.thumbnail[i];
            sum += abs(diff.r) + abs(diff.g) + abs(diff.b);
        }

        return 1.0f / sum;
    }

    ImageEntry computeEntry(ImageData img)
    {
        ImageEntry entry;
        entry.mean = img.mean;
        entry.std_deviation = img.std_deviation;
        stbir_resize_float(&img.pixels[0].r, img.width, img.height, 0, &entry.thumbnail[0].r, ThumbnailSize, ThumbnailSize, 0, 3);
        return entry;
    }

    ImageEntry computeEntry(ImageBlockView img)
    {
        ImageEntry entry{};
        for (int j = 0; j < img.height; ++j)
        {
            for (int i = 0; i < img.width; ++i)
            {
                entry.mean += img(i, j);
                entry.std_deviation += img(i, j) * img(i, j);
            }
        }
        entry.mean /= img.width * img.height;
        entry.std_deviation = entry.std_deviation / (img.width * img.height) - entry.mean * entry.mean;
        stbir_resize_float(&img(0, 0).r, img.width, img.height, img.img->width * sizeof(RgbPixel), &entry.thumbnail[0].r, ThumbnailSize, ThumbnailSize, 0, 3);
        return entry;
    }

    std::vector<ImageEntry> loadEntries(const std::filesystem::path &db_folder)
    {
        using std::begin, std::end;

        auto index_path = db_folder.lexically_normal();
        if (!index_path.has_filename())
            index_path = index_path.parent_path();
        index_path += ".imgdb";
        if (std::filesystem::is_regular_file(index_path))
        {
            std::ifstream index(index_path, std::ios_base::binary);
            return rawVectorLoad<ImageEntry>(index);
        }
        //No premade index, build it
        OrderedDirectory dir(db_folder);
        std::vector<ImageEntry> result(dir.size());
        auto entry_it = result.begin();
        std::transform(std::execution::par_unseq, begin(dir), end(dir), begin(result), [&](auto& path)
        {
            return computeEntry(loadImage(path.string().c_str()));
        });
        std::ofstream index(index_path, std::ios_base::binary);
        rawVectorStore(index, result);

        return result;
    }

    float calcEntriesScore(const ImageEntry &l, const ImageEntry &r)
    {
        /*using std::abs;
        RgbPixel diff = l.mean - r.mean;
        return -abs(diff.r) - abs(diff.g) - abs(diff.b);*/
        return computePSNR(l, r);
    }
}

RgbPixel &ImageBlockView::operator()(int i, int j)
{
    return img->pixels[x_start + i + (y_start + j) * img->width];
}

OrderedDirectory::OrderedDirectory(const std::filesystem::path &directory)
    : _files(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{})
{
    if (!std::filesystem::is_directory(directory))
        throw std::invalid_argument("Not a folder");
    std::ranges::sort(_files);
}

std::size_t OrderedDirectory::size() const
{
    return _files.size();
}

const std::filesystem::path &OrderedDirectory::operator[](std::size_t i)
{
    return _files[i];
}

const RgbPixel &ImageEntry::operator()(std::size_t i, std::size_t j) const
{
    return thumbnail[j * ThumbnailSize + i];
}

RgbPixel &ImageEntry::operator()(std::size_t i, std::size_t j)
{
    return const_cast<RgbPixel &>(std::as_const(*this)(i, j));
}

ImageDatabase::ImageDatabase(const std::filesystem::path &db_folder)
    : _entries(loadEntries(db_folder)), _used(std::make_unique<std::atomic_bool[]>(_entries.size()))
{
	for(std::size_t i = 0; i < _entries.size(); ++i) std::atomic_init(&_used[i], false);
}

std::size_t ImageDatabase::findBestEntry(ImageBlockView block) const
{
    ImageEntry img = computeEntry(block);

    return doFindBestEntry(img);
}

std::size_t ImageDatabase::findBestEntryUnique(ImageBlockView block)
{
	ImageEntry img = computeEntry(block);
	std::size_t entry;

	do
	{
		entry = doFindBestEntry(img);
	} while(_used[entry].exchange(true, std::memory_order_relaxed));

	return entry;
}

std::size_t ImageDatabase::size() const
{
	return _entries.size();
}

std::size_t ImageDatabase::doFindBestEntry(const ImageEntry& img) const
{
    std::size_t best_i = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < _entries.size(); ++i)
    {
        if (float score = calcEntriesScore(img, _entries[i]); score > best_score && !_used[i].load(std::memory_order_relaxed))
        {
            best_i = i;
            best_score = score;
        }
    }

    return best_i;
}