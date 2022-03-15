#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <vector>
#include <iostream>

#include <stb_image.h>
#include <stb_image_write.h>

struct RgbPixel
{
    float r, g, b;
};

struct ImageData
{
    std::vector<RgbPixel> pixels;
    RgbPixel mean;
    RgbPixel std_deviation;

    int height, width;
};

std::ostream &operator<<(std::ostream &out, const RgbPixel &rightHand);
RgbPixel &operator+=(RgbPixel &leftHand, const RgbPixel &rightHand);
RgbPixel &operator-=(RgbPixel &leftHand, const RgbPixel &rightHand);
RgbPixel &operator/=(RgbPixel &leftHand, const float factor);
RgbPixel &operator*=(RgbPixel &leftHand, const RgbPixel &rightHand);
RgbPixel operator+(RgbPixel leftHand, const RgbPixel &rightHand);
RgbPixel operator-(RgbPixel leftHand, const RgbPixel &rightHand);
RgbPixel operator/(RgbPixel leftHand, const float factor);
RgbPixel operator*(RgbPixel leftHand, const RgbPixel &rightHand);

ImageData loadImage(const char *path);
void saveImage(const ImageData &data, const char* path);

#endif