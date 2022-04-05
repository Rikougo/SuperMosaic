#include <cstdint>

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

__global__ void deviceSelectEntries(RgbPixel* pixels, int height, int width, ImageEntry* entries, std::size_t n_entries, std::size_t* blocks, int block_size)
{
	__shared__ float block_thumbnail[blockDim.y*blockDim.x];

	int thumbnailIndex = threadIdx.y * blockDim.x + threadIdx.x;
	float u_coord = (float)threadIdx.x / blockDim.x * block_size + block_size * blockIdx.x;
	float v_coord = (float)threadIdx.y / blockDim.y * block_size + block_size * blockIdx.y;
	block_thumbnail[thumbnailIndex] = sampleLinear(pixels, height, width, u_coord, v_coord);
	__syncthreads();
}

extern "C" void cudaSelectEntries(RgbPixel* pixels, int height, int width, ImageEntry* entries, std::size_t n_entries, std::size_t* blocks, int block_size)
{
}
