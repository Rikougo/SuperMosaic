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

__device__ float sqrPixelDiff(RgbPixel l, RgbPixel r)
{
    return (l.r - r.r) * (l.r - r.r) + (l.g - r.g) * (l.g - r.g) + (l.b - r.b) * (l.b - r.b);
}

__global__ void deviceSelectEntries(RgbPixel* pixels, int height, int width, ImageEntry* entries, unsigned* entries_usage, std::size_t n_entries, std::size_t* blocks, int block_size)
{
	//__shared__ RgbPixel block_thumbnail[blockDim.x*blockDim.y];
    __shared__ float sum = 0;
    __shared__ std::size_t best_entry = -1;
    __shared__ float best_score = 0;

	int thumbnailIndex = threadIdx.y * blockDim.x + threadIdx.x;
	float u_coord = (float)threadIdx.x / blockDim.x * block_size + block_size * blockIdx.x;
	float v_coord = (float)threadIdx.y / blockDim.y * block_size + block_size * blockIdx.y;
	//block_thumbnail[thumbnailIndex] = sampleLinear(pixels, height, width, u_coord, v_coord);
    RgbPixel pixel = sampleLinear(pixels, height, width, u_coord, v_coord);
	__syncthreads();
    
    while(best_entry == -1)
    {
        for(std::size_t i = 0; i < n_entries; ++i)
        {
            atomicAdd_block(&sum, sqrPixelDiff(pixel, entries[i].thumbnail[thumbnailIndex]));
            __syncthreads();
            if(thumbnailIndex == 0)
            {
                float score = 1.0f/sum;
                if(score > best_score && entries_usage[i] == 0)
                {
                    best_entry = i;
                    best_score = score;
                }
                sum = 0;
            }
            __syncthreads();
        }
        if(thumbnailIndex == 0)
        {
            if(atomicCAS(&entries_usage[best_entry], 0, 1) == 1)
            {
                best_entry = -1;
                best_score = 0;
            }
        }
        __syncthreads();
    }
    
    if(thumbnailIndex == 0)
    {
        blocks[blockIdx.y * gridDim.x + blockDim.x] = best_entry;
    }
}

extern "C" void cudaSelectEntries(RgbPixel* pixels, int height, int width, ImageEntry* entries, std::size_t n_entries, std::size_t* blocks, int block_size)
{
    int x_blocks = width / block_size;
    int y_blocks = height / block_size;
    
    RgbPixel* dpixels;
    ImageEntry* dentries;
    unsigned* dentries_usage;
    std::size_t* dblocks;
    
    cudaMalloc(&dpixels, height*width*sizeof(RgbPixel));
    cudaMemcpy(dpixels, pixels, height*width*sizeof(RgbPixel), cudaMemcpyHostToDevice);
    cudaMalloc(&dentries, n_entries*sizeof(ImageEntry));
    cudaMemcpy(dentries, entries, n_entries*sizeof(ImageEntry), cudaMemcpyHostToDevice);
    cudaMalloc(&dentries_usage, n_entries*sizeof(unsigned));
    cudaMemset(dentries_usage, 0, n_entries*sizeof(unsigned));
    cudaMalloc(&dblocks, x_blocks*y_blocks*sizeof(std::size_t));
    
    dim3 block_dim(ThumbnailSize,ThumbnailSize,1);
    dim3 grid_dim(x_blocks,y_blocks,1);
    deviceSelectEntries<<<grid_dim, block_dim>>>(dpixels, height, width, dentries, dentries_usage, n_entries, dblocks, block_size);
    cudaMemcpy(blocks, dblocks, x_blocks*y_blocks*sizeof(std::size_t), cudaMemcpyDeviceToHost);
    
    cudaFree(dpixels);
    cudaFree(dentries);
    cudaFree(dentries_usage);
    cudaFree(dblocks);
}
