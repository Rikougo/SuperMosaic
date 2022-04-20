#include <image_loader.h>
#include <string>
#include <thread>

std::ostream &operator<<(std::ostream &out, const RgbPixel &rightHand)
{
    return out << "(" << rightHand.r << ", " << rightHand.g << ", " << rightHand.b << ")";
}

RgbPixel &operator+=(RgbPixel &leftHand, const RgbPixel &rightHand)
{
    leftHand.r += rightHand.r;
    leftHand.b += rightHand.b;
    leftHand.g += rightHand.g;
    return leftHand;
}

RgbPixel &operator-=(RgbPixel &leftHand, const RgbPixel &rightHand)
{
    leftHand.r -= rightHand.r;
    leftHand.b -= rightHand.b;
    leftHand.g -= rightHand.g;
    return leftHand;
}

RgbPixel &operator/=(RgbPixel &leftHand, const float factor)
{
    leftHand.r /= factor;
    leftHand.b /= factor;
    leftHand.g /= factor;
    return leftHand;
}

RgbPixel &operator*=(RgbPixel &leftHand, const RgbPixel &rightHand)
{
    leftHand.r *= rightHand.r;
    leftHand.b *= rightHand.b;
    leftHand.g *= rightHand.g;
    return leftHand;
}

RgbPixel operator+(RgbPixel leftHand, const RgbPixel &rightHand)
{
    return leftHand += rightHand;
}

RgbPixel operator-(RgbPixel leftHand, const RgbPixel &rightHand)
{
    return leftHand -= rightHand;
}

RgbPixel operator/(RgbPixel leftHand, const float factor)
{
    return leftHand /= factor;
}

RgbPixel operator*(RgbPixel leftHand, const RgbPixel &rightHand)
{
    return leftHand *= rightHand;
}

ImageData loadImage(const char *path)
{
    using namespace std::string_literals;
    ImageData imgData;
    int channels;

    unsigned char* data = nullptr;
    while(true)
    {
        data = stbi_load(path, &imgData.width, &imgData.height, &channels, 3);
        if (data) break;
        std::this_thread::yield();
    }
    int size = imgData.width * imgData.height;
    imgData.pixels.resize(size);

    imgData.mean = imgData.std_deviation = {0, 0, 0};

    for (int i = 0; i < imgData.height; i++)
    {
        for (int j = 0; j < imgData.width; j++)
        {
            auto pos = i * imgData.width + j;
            imgData.pixels[pos].r = data[pos * 3] / 255.0f;
            imgData.pixels[pos].g = data[pos * 3 + 1] / 255.0f;
            imgData.pixels[pos].b = data[pos * 3 + 2] / 255.0f;

            imgData.mean += imgData.pixels[pos];
        }
    }

    imgData.mean /= (imgData.width * imgData.height);

    for (auto const &v : imgData.pixels)
    {
        auto distance = v - imgData.mean;
        imgData.std_deviation += distance * distance;
    }

    imgData.std_deviation /= (imgData.width * imgData.height);

    stbi_image_free(data);

    return imgData;
}

void saveImage(const ImageData &data, const char* path) {
    unsigned char* values = new unsigned char[data.width * data.height * 3];

    for (int i = 0; i < data.height; i++)
    {
        for (int j = 0; j < data.width; j++)
        {
            int pos = i * data.width + j;
            values[pos*3  ] = data.pixels[pos].r * 255;
            values[pos*3+1] = data.pixels[pos].g * 255;
            values[pos*3+2] = data.pixels[pos].b * 255;
        }
    }
    stbi_write_png(path, data.width, data.height, 3, values, 0);

    delete[] values;
}