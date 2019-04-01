#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

// Inverse
__global__ void paraInverse(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels);

//Brightness
__global__ void paraBrightness(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels, int shift);

//parFlipV
__global__ void parFlipV(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//parFlipH
__global__ void parFlipH(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

__global__ void parRotateAnti(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

__global__ void parRotateClock(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

__global__ void parCrop(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels, int x1, int y1, int x2, int y2);

__global__ void parNot(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

__global__ void parAnd(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels);

__global__ void parOr(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels);


__global__ void parXor(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels);

__global__ void par_mean_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

__global__ void par_gaussian_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

#endif
