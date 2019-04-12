#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>
#include "cudaImage.h"

#define TILE_WIDTH 16

using namespace std;

void seqInverse(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < dataSizeY; ++i)                //cycle on image rows
        {
            for (j = 0; j < dataSizeX; ++j)            //cycle on image columns
            {
               outputImageData[(dataSizeX * i + j)*channels + k] =255- inputImageData[(dataSizeX * i + j)*channels + k];
            }
        }
    }
}

void seqBrightness(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels,int shift)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
	int value;
    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < dataSizeY; ++i)                //cycle on image rows
        {
            for (j = 0; j < dataSizeX; ++j)            //cycle on image columns
            {
              
				value	= inputImageData[(dataSizeX * i + j)*channels + k]+shift;

				if( value >255)
					{
						value=255;
					}
				else if ( value<0)
					{
						value=0;
					}
				
				outputImageData[(dataSizeX * i +j)*channels +k]= value;
					
            }
        }
    }
}


void seqFlipV(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
     int i, j;
    const unsigned char * inputImageData = inputImage;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width; ++j)            //cycle on image columns
            {
                outputImageData[( i*width + j)*channels + k] = inputImageData[(( i * width) + (width - 1- j))*channels + k];
            }
        }
    }
}

void seqFlipH(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
     int i, j;
     const unsigned char *inputImageData = inputImage;

      for(int k=0; k<channels; k++) {
                for (i = 0; i<height; i++)
	{
		for(j = 0; j < width; j++) 
		{
			outputImageData[( i *width + j)*channels + k] = inputImageData[(( height - 1 -i ) * width + j) * channels +k];
		}
	}
    }
}


void seqRotateAnti(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData[( i * width + j)*channels + k] = inputImageData[((  j * width) + (width - 1 - i ))*channels + k];
            }
        }
    }
}


void seqRotateClock(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData[( i *width + j)*channels + k] = inputImageData[((height - 1 - j) * width + i )*channels + k];
            }
        }
    }
}

void seqCrop(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height ,int channels, int x1,int y1,int x2,int y2)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
   int new_h=x2-x1+1;
   int new_w=y2-y1+1;
    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i <new_h ; ++i)                //cycle on image rows
        {
            for (j = 0; j < new_w ; ++j)            //cycle on image columns
            {
                outputImageData[( i * new_w + j)*channels + k] = inputImageData[((x1+i)*width+(y1+j))*channels+k];
            }
        }
    }
}

void seqNot(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData[( i *width + j)*channels + k] = 255 - inputImageData[(i *width + j )*channels + k];
            }
        }
    }
}

void seqAnd(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels)
{
    int i, j;
    const unsigned char * inputImageData1 = inputImage1;

    const unsigned char * inputImageData2 = inputImage2;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData[( i *width + j)*channels + k] =  inputImageData1[( i *width + j)*channels + k] & inputImageData2[( i *width + j)*channels + k];
            }
        }
    }
}

void seqOr(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels)
{
    int i, j;
    const unsigned char * inputImageData1 = inputImage1;

    const unsigned char * inputImageData2 = inputImage2;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData[( i *width + j)*channels + k] =  inputImageData1[( i *width + j)*channels + k] | inputImageData2[( i *width + j)*channels + k];
            }
        }
    }
}

void seqXor(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels)
{
    int i, j;
    const unsigned char * inputImageData1 = inputImage1;

    const unsigned char * inputImageData2 = inputImage2;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < height; ++i)                //cycle on image rows
        {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData[( i *width + j)*channels + k] =  inputImageData1[( i *width + j)*channels + k] ^ inputImageData2[( i *width + j)*channels + k];
            }
        }
    }
}

void seq_mean_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float temp=0;
			
			    for (int i1=-1;i1<=1;i1++)
				{
					for (int j1=-1;j1<=1;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) continue;
						 temp=temp+inputImageData[(((i+i1)*width+j+j1)*channels)+k];
					}
				}
			
                outputImageData[( i *width + j)*channels + k] = (int) (temp/9.0);
            }
        }
    }
}


void seq_gaussian_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   const unsigned char gaus[25]={1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};

   
    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float temp=0;
			
			    for (int i1=-2;i1<=2;i1++)
				{
					for (int j1=-2;j1<=2;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) continue;
						 temp=temp+inputImageData[(((i+i1)*width+j+j1)*channels)+k]*gaus[(i1+2)*5+j1+2];
					}
				}
			
                outputImageData[( i *width + j)*channels + k] = (int) (temp/273.0);
            }
        }
    }
 }


int main()
{
	int channels;
	
    // Set Channel Value;
	channels= 3;
	
	int width, height,width1,height1,width2,height2, bpp,bpp1, bpp2;
   	unsigned char * sequential, *sequential2, *seq_img_crop;
	const unsigned char* image, *image1, *image2;
	float runTime;
		
    /*image = stbi_load( "img.png", &width, &height, &bpp, channels );
    image1= stbi_load( "img1.png", &width1, &height1, &bpp1, channels );
	image2= stbi_load( "img2.jpg", &width2, &height2, &bpp2, channels );*/
	
	image = stbi_load( "lena128.png", &width, &height, &bpp, channels );
    image1= stbi_load( "lenaFlip128.png", &width1, &height1, &bpp1, channels );
	image2= stbi_load( "lena128.png", &width2, &height2, &bpp2, channels );
	
    sequential = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));
	sequential2 = (unsigned char*)malloc(width2*height2*channels*sizeof(unsigned char));
		
   	cout <<"SEQUENTIAL" << endl;
	//cout << "image dimensions: "<< width << "x" << height << endl;

	//Inverse
	cout << "Inverse elapsed in time: ";
	 clock_t begin_time = clock();	
	seqInverse(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) / (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms"<<endl;
	stbi_write_png("Inverse.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Brightness
	cout << "Brightness elapsed in time: ";
    begin_time = clock();   
	seqBrightness(image, sequential, width, height, channels,-128); 
	runTime = (float)( clock() - begin_time ) / ( CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("Brightness.png", width, height, channels, sequential, 0);	
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
		
	//Vertical flip
	cout << "seqFlipV elapsed in time: ";
    begin_time = clock();   
	seqFlipV(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("FlipVertical.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Horizontal flip
	cout << "seqFlipH elapsed in time: ";
    begin_time = clock();   
	seqFlipH(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("FlipHorizontal.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Anticlockwise rotation
	cout << "seqRotateAnti elapsed in time: ";
    begin_time = clock();   
	seqRotateAnti(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("AntiClockRotation.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Clockwise rotation
	cout << "seqRotateClock elapsed in time: ";
    begin_time = clock();   
	seqRotateClock(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("ClockRotation.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//And
	if(width==width1 && height==height1) 
        {
	    cout << "seqAnd elapsed in time: ";
        begin_time = clock();   
	
	 seqAnd(image,image1, sequential, width, height, channels);
	 runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	 cout<< runTime <<" ms" <<endl;
	 stbi_write_png("AndOperation.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	}
	else
	{
	 cout<< "ERROR! Images should be of same size for this Operation";
	}
	//OR
	if(width==width1 && height==height1) 
        {
	cout << "seqOR elapsed in time: ";
    	begin_time = clock(); 
		
	seqOr(image,image1, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("OROperation.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	}
	else
	{
	 cout<< "ERROR! Images should be of same size for this Operation";
	}
	//XOR
	if(width==width1 && height==height1) 
        {
	cout << "seqXOR elapsed in time: ";
    begin_time = clock();   
	seqXor(image,image1, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("XOROperation.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	}
	else
	{
	 cout<< "ERROR! Images should be of same size for this Operation";
	}
	
	//NOT
	cout << "seqNOT elapsed in time: ";
    begin_time = clock();   
	seqNot(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("NOTOperation.png", width, height, channels, sequential, 0);
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Crop
	cout << "CropOperation elapsed in time: ";
    int x1= 50;
    int y1= 50;
    int x2= 100;
    int y2= 100; 
	begin_time = clock();
	seq_img_crop = (unsigned char*)malloc((y2-y1+1)*(x2-x1+1)*sizeof(unsigned char)*channels);
	seqCrop(image, seq_img_crop, width, height, channels,x1,y1,x2,y2);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("CropOperation.png", (y2-y1+1), (x2-x1+1), channels, seq_img_crop, 0);
	
	memset(sequential2,0,sizeof(width2*height2*channels*sizeof(unsigned char)));
	
	// Mean Filter
	cout << "Mean Filter elapsed in time: ";
    begin_time = clock();   
	seq_mean_filter(image2, sequential2, width2, height2, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("MeanFilter.png", width2, height2, channels, sequential2, 0);
		
	memset(sequential2,0,sizeof(width2*height2*channels*sizeof(unsigned char)));
	
	// Gaussian Filter
	cout << "Gaussian Filter elapsed in time: ";
    begin_time = clock();   
	seq_gaussian_filter(image2, sequential2, width2, height2, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("GaussianFilter.png", width2, height2, channels, sequential2, 0);
	

	cout << "*----------------------------------*" << endl;

/******************************************************************************************************************/	
	
	cudaEvent_t startEvent,stopEvent;
		
	unsigned char *deviceInputImageData,*deviceInputImageData1, *deviceInputImageData2;
    unsigned char *deviceOutputImageData, *deviceOutputImageData1, *deviceOutputImageData2;
    runTime=0.0;
	cudaDeviceReset();
    cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);	
	unsigned char *Parallel = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));
	unsigned char *Parallel2 = (unsigned char*)malloc(width2*height2*channels*sizeof(unsigned char));
	unsigned char *Para_crop;
	
	cudaMalloc((void **) &deviceInputImageData, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageData1, width1 * height1 *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageData2, width2 * height2 *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageData, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageData1, (y2-y1+1)*(x2-x1+1) * channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageData2, width2 * height2 *channels * sizeof(unsigned char));
	cudaMemcpy(deviceInputImageData, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageData1, image1, width1 * height1 * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageData2, image2, width2 * height2 * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	dim3 dimGrid(ceil((float) width/TILE_WIDTH), ceil((float) height/TILE_WIDTH));
	dim3 dimGrid2(ceil((float) width2/TILE_WIDTH), ceil((float) height2/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	
	cout <<"PARALLEL" << endl;
	cout << "image dimensions: "<< width << "x" << height << endl;
	
	//Inverse
	cout << "Inverse elapsed in time: ";
	cudaEventRecord(startEvent);
	paraInverse<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraInverse.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Brightness
	cout << "Brightness elapsed in time: ";
	cudaEventRecord(startEvent);
	paraBrightness<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels,-128);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraBrightness.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
		
	//parFlipV
	cout << "parFlipV elapsed in time: ";
	cudaEventRecord(startEvent);
	parFlipV<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parFlipV.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//parFlipH
	cout << "parFlipH elapsed in time: ";
	cudaEventRecord(startEvent);
	parFlipH<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parFlipH.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//parRotateAnti
	cout << "parRotateAnti elapsed in time: ";
	cudaEventRecord(startEvent);
	parRotateAnti<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parRotateAnti.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//parRotateClock
	cout << "parRotateClock elapsed in time: ";
	cudaEventRecord(startEvent);
	parRotateClock<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parRotateClock.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//parAnd
	if(width==width1 && height==height1) 
        {
	cout << "parAnd elapsed in time: ";
	cudaEventRecord(startEvent);
	parAnd<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceInputImageData1, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parAnd.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	}
	else
	{
	 cout<< "ERROR! Images should be of same size for this Operation";
	}
	
	//parOr
	cout << "parOr elapsed in time: ";
	cudaEventRecord(startEvent);
	parOr<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceInputImageData1, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parOr.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//parXor
	if(width==width1 && height==height1) 
        {
	cout << "parXor elapsed in time: ";
	cudaEventRecord(startEvent);
	parXor<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceInputImageData1, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parXor.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	}
	else
	{
	 cout<< "ERROR! Images should be of same size for this Operation";
	}
	
	//parNot
	if(width==width1 && height==height1) 
        {
	cout << "parNot elapsed in time: ";
	cudaEventRecord(startEvent);
	parNot<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parNot.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	}
	else
	{
	 cout<< "ERROR! Images should be of same size for this Operation";
	}
	
	//parCrop
	cout << "parCrop elapsed in time: ";
	cudaEventRecord(startEvent);
	Para_crop = (unsigned char*)malloc((y2-y1+1)*(x2-x1+1)*sizeof(unsigned char)*channels);
	parCrop<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData1, width, height, channels, x1,y1,x2,y2);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Para_crop, deviceOutputImageData1, (y2-y1+1)*(x2-x1+1)* channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parCrop.png", (y2-y1+1), (x2-x1+1), channels, Para_crop, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Gaussian filter
	cout << "GaussianFilter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_gaussian_filter<<<dimGrid2,dimBlock>>>(deviceInputImageData2, deviceOutputImageData2, width2, height2, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel2, deviceOutputImageData2, width2 * height2 * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraGaussianFilter.png", width2, height2, channels, Parallel2, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Mean filter
	cout << "MeanFilter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_mean_filter<<<dimGrid2,dimBlock>>>(deviceInputImageData2, deviceOutputImageData2, width2, height2, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel2, deviceOutputImageData2, width2 * height2 * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraMeanFilter.png", width2, height2, channels, Parallel2, 0); 
	
    return 0;
}
