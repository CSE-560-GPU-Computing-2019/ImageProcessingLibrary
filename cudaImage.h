#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

// Inverse
__global__ void paraInverse(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels);

//Brightness
__global__ void paraBrightness(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels, int shift);

//FlipV
__global__ void parFlipV(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//FlipH
__global__ void parFlipH(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Rotate Anti-Clockwise
__global__ void parRotateAnti(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Rotate Clockwise
__global__ void parRotateClock(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Crop
__global__ void parCrop(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels, int x1, int y1, int x2, int y2);

//Not
__global__ void parNot(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//And
__global__ void parAnd(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels);

//Or
__global__ void parOr(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels);

//Xor
__global__ void parXor(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels);

//Mean Filter
__global__ void par_mean_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Gaussian Filter
__global__ void par_gaussian_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Median Filter kernel 3
__global__ void par_median_filter3(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Median Filter kernel 5
__global__ void par_median_filter5(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Max Filter
__global__ void par_max_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Min Filter
__global__ void par_min_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Edge Detection
__global__ void par_edge_detection(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels);

//Channel Split
__global__ void parchannelSplit(const unsigned char *inputImage1, unsigned char *outputImageData_Red,unsigned char *outputImageData_Green,unsigned char *outputImageData_Blue, int width, int height);

//RGB to GrayScale image
__global__ void parRGBToGrey(const unsigned char *inputImage1, unsigned char *outputImageData, int width, int height);

//GrayScale to binary image
__global__ void parGreyToBinary(const unsigned char *inputImage1, unsigned char *outputImageData, int width, int height, float level);

//Adaptive Noise Reduction Filter
__global__ void par_adp_local_noise_reduction_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,int channels, int *sum_total, int *sum_square_total);

//Adaptive Median Filter
__global__ void par_adp_median_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,int channels);

//Dialation Binary
__global__ void parDialation_binary(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height);

//Erode Binary
__global__ void parErode_binary(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height);

//Dialation GrayScale
__global__ void parDialation_Grey(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height);

//Erode GrayScale
__global__ void parErode_Grey(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height);


#endif
