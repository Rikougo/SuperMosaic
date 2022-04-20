#include <cstddef>

struct RgbPixel
{
    float r, g, b;
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

__device__ float lerp(float l, float r, float t)
{
	return l + t * (r-l);
}

__device__ RgbPixel lerp(RgbPixel l, RgbPixel r, float t)
{
	return RgbPixel{lerp(l.r, r.r, t),lerp(l.g, r.g, t),lerp(l.b, r.b, t)};
}

__device__ RgbPixel sampleLinear(RgbPixel* pixels, int height, int width, float uf, float vf)
{
	int u = (int)uf;
	int v = (int)vf;
	int u_next = min(u + 1, width - 1);
	int v_next = min(v + 1, height - 1);
	return lerp(lerp(pixels[v*width+u], pixels[v*width+u_next], uf - u), lerp(pixels[v_next*width+u], pixels[v_next*width+u_next], uf - u), vf - v);
}

__device__ float sqrPixelDiff(RgbPixel l, RgbPixel r)
{
    return (l.r - r.r) * (l.r - r.r) + (l.g - r.g) * (l.g - r.g) + (l.b - r.b) * (l.b - r.b);
}

__global__ void deviceCalcScoreMatrix(RgbPixel* pixels, int height, int width, ImageEntry* entries, std::size_t n_entries, int block_size, float* matrix)
{
    std::size_t x_blocks = width / block_size;
    std::size_t y_blocks = height / block_size;
    std::size_t n_blocks = x_blocks * y_blocks;
    std::size_t block_id = threadIdx.x + blockDim.x * blockIdx.x;
    std::size_t entry_id = threadIdx.y + blockDim.y * blockIdx.y;
    if (block_id >= n_blocks || entry_id >= n_entries) return;

    std::size_t block_x = block_id % x_blocks;
    std::size_t block_y = block_id / x_blocks;

    ImageEntry entry = entries[entry_id];
    float sum = 0;
    for (int j = 0; j < ThumbnailSize; ++j)
    {
        float v_coord = (float)j / ThumbnailSize * block_size + block_size * block_y;
        for (int i = 0; i < ThumbnailSize; ++i)
        {
            float u_coord = (float)i / ThumbnailSize * block_size + block_size * block_x;
            RgbPixel pixel = sampleLinear(pixels, height, width, u_coord, v_coord);
            sum += sqrPixelDiff(pixel, entry.thumbnail[i + j * ThumbnailSize]);
        }
    }
    matrix[entry_id + block_id * n_entries] = 1.0f / sum;
}

__global__ void deviceSelectEntries(float* matrix, unsigned* entries_usage, std::size_t n_entries, std::size_t* blocks, std::size_t n_blocks)
{
    std::size_t block_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (block_id >= n_blocks) return;

    std::size_t best_entry = n_entries;
    float best_score = 0;

    while (best_entry == n_entries)
    {
        for (std::size_t i = 0; i < n_entries; ++i)
        {
            float score = matrix[i + block_id * n_entries];
            if (score > best_score && entries_usage[i] == 0)
            {
                best_entry = i;
                best_score = score;
            }
        }
        if (atomicExch(&entries_usage[best_entry], 1) == 1)
        {
            best_entry = n_entries;
            best_score = 0;
        }
    }
    blocks[block_id] = best_entry;
}

extern "C" void cudaSelectEntries(const RgbPixel* pixels, int height, int width, ImageEntry* entries, std::size_t n_entries, std::size_t* blocks, int block_size)
{
    std::size_t x_blocks = width / block_size;
    std::size_t y_blocks = height / block_size;
    std::size_t n_blocks = x_blocks * y_blocks;

    RgbPixel* dpixels;
    ImageEntry* dentries;
    unsigned* dentries_usage;
    std::size_t* dblocks;
    float* dmatrix;

    cudaMalloc(&dpixels, height * width * sizeof(RgbPixel));
    cudaMemcpy(dpixels, pixels, height * width * sizeof(RgbPixel), cudaMemcpyHostToDevice);
    cudaMalloc(&dentries, n_entries * sizeof(ImageEntry));
    cudaMemcpy(dentries, entries, n_entries * sizeof(ImageEntry), cudaMemcpyHostToDevice);
    cudaMalloc(&dentries_usage, n_entries * sizeof(unsigned));
    cudaMemset(dentries_usage, 0, n_entries * sizeof(unsigned));
    cudaMalloc(&dmatrix, n_blocks * n_entries * sizeof(float));
    cudaMalloc(&dblocks, n_blocks * sizeof(std::size_t));

    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((n_blocks + 15) / 16, (n_entries + 15) / 16, 1);
    deviceCalcScoreMatrix<<<grid_dim, block_dim>>>(dpixels, height, width, dentries, n_entries, block_size, dmatrix);
    dim3 block_dim2(16, 1, 1);
    dim3 grid_dim2((n_blocks + 15) / 16, 1, 1);
    deviceSelectEntries<<<grid_dim2, block_dim2>>>(dmatrix, dentries_usage, n_entries, dblocks, n_blocks);

    cudaMemcpy(blocks, dblocks, n_blocks * sizeof(std::size_t), cudaMemcpyDeviceToHost);

    cudaFree(dpixels);
    cudaFree(dentries);
    cudaFree(dentries_usage);
    cudaFree(dblocks);
    cudaFree(dmatrix);
    //*/
}
