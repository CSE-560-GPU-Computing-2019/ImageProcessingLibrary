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

#define imgchannels 1

#define TILE_WIDTH 16
#define w (TILE_WIDTH + maskCols -1)
#define KERNEL_RADIUS 1

using namespace std;

void seq_hole_filling_1(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    unsigned char *X=new unsigned char[width*height];
	unsigned char *X1=new unsigned char[width*height];
	const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={0,1,0,1,1,1,0,1,0};
    int rad=1;
    memset((void *) X,0,width*height*sizeof(unsigned char));
    
	X[208*width+232]=255;
	
	// finding complement of inputImage
	unsigned char *input_c = new unsigned char[width*height];
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
		    int temp=inputImage[i*width+j];
			if (temp==255)
			     input_c[i*width+j]=0;
			else
			     input_c[i*width+j]=255;
		}
    int flag=0;
	int count=0;
	while(flag==0)
	{
	    // dialation of X and result is stored in X1  
		for (int i = 0; i < height; ++i)                //cycle on image rows
		{
			for (int j = 0; j < width ; ++j)            //cycle on image columns
			{	  
				int flag1=0;
				for(int ki=-rad; ki<=rad && flag==0; ki++)
				{
					for(int kj=-rad;kj<=rad;kj++)
					{
						if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width) continue;
						if ( B[rad+ki][rad+kj]==1 && X[((i+ki)*width + j+kj)]==255) 
						{		
							flag1=1;
							break;
						}
					}
				}
				if (flag1==1)
					    X1[i *width + j] =  255;
				else
					    X1[i *width + j] =  0;
            }
        }

			
		// Intersection with A compliment

		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				int temp=i*width+j;
				if (X1[temp]==255 && input_c[temp]==255)
					X1[temp]=255;
				else
	                X1[temp]=0;
			}
		}
		flag=1;
		for(int i=0;i<height && flag==1;i++)
		{
			for(int j=0;j<width;j++)
			{
				int temp=i*width+j;
				if (X1[temp]!=X[temp]) 
				{ 
				   flag=0;
				   break;
				}	
			}
		}
		
		if (flag==0)
		{
			for(int i=0;i<height;i++)
			{
				for(int j=0;j<width;j++)
				{
					int temp=i*width+j;
					X[temp]=X1[temp];
					
				}	
			}
		}
	
	}
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			int temp=i*width+j;
			if (X1[temp]==255 || inputImage[temp]==255) outputImageData[temp]=255;
            else outputImageData[temp]=0;			
		}
	}
}

__global__ void par_hole_filling(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,unsigned char* X, unsigned char *X1,int *flag)
{
     unsigned char B[3][3]={0,1,0,1,1,1,0,1,0};
    int rad=1;
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
    
	int temp=inputImage[i*width+j];
	
	// every thread will calculate its compliment and store it in outImageDtata. calculating compliment of imputImage and storing it in outputImageData
	X[208*width+232]=255;
	if (temp==255) outputImageData[i*width+j]=0;
	else outputImageData[i*width+j]=255;
	
	if (i==0) *flag=1;
	__syncthreads();
    
	// outputImageData contains compliment of inputImage
	
	int count=0;
	
	while(*flag>0)
	{
	    	
		// dialation of X
		int flag1=0;
		for(int ki=-rad; ki<=rad && flag1==0; ki++)
		{
			for(int kj=-rad; kj<=rad; kj++)
			{
				if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width) continue;
				if ( B[rad+ki][rad+kj]==1 && X[((i+ki)*width + j+kj)]==255) 
				{		
						flag1=1;
						break;
				}	
			}
		}
		
		if (flag1==1)
			X1[i *width + j] =  255;
		else
			X1[i *width + j] =  0;

		// Intersection with A compliment

		temp=i*width+j;
		
		if (X1[temp]==255 && outputImageData[temp]==255)
					X1[temp]=255;
		else
	                X1[temp]=0;
		
		if (i==0) 	{ *flag=0;}
		__syncthreads();
		
	    if (X1[temp]!=X[temp]) 
				atomicAdd(flag,1);
		
		__syncthreads();
		
		if (*flag>0)
		{
			X[temp]=X1[temp];
		}
		__syncthreads();
	
	}
	if (X1[temp]==255 || inputImage[temp]==255) outputImageData[temp]=255;
    else outputImageData[temp]=0;
}


int main()
{
	int channels;
	
    // Set Channel Value;
	channels= 1;
	
	int width, height, bpp;
   	unsigned char *sequential;
	const unsigned char *image;
	float runTime;
		
	image = stbi_load( "imgg.png", &width, &height, &bpp, channels );
	
    sequential = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));
    	
   	cout <<"SEQUENTIAL" << endl;
	cout<<width<<" "<<height<<endl;
	cout << "image dimensions: "<< width << "x" << height << endl;
    
	// Hole filling
	cout << "Hole Filling elapsed in time: ";
    clock_t begin_time = clock();   
	seq_hole_filling_1(image, sequential, width, height);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("seq_hole_filling.png", width, height, channels, sequential, 0);
	
	cout << "*----------------------------------*" << endl;

/******************************************************************************************************************/	

	cudaEvent_t startEvent,stopEvent;
		
	unsigned char *deviceInputImageData;
    unsigned char *deviceOutputImageData;
	unsigned char *device_X, *device_X1;
	int *device_flag=0;
	
    runTime=0.0;
	cudaDeviceReset();
    cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);	
	unsigned char *Parallel = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));
	
	cudaMalloc((void **) &deviceInputImageData, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageData, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &device_X, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &device_X1, width * height *channels * sizeof(unsigned char));
	
	cudaMemset(device_X,0,sizeof(width*height*channels*sizeof(unsigned char)));
	cudaMemset(device_X1,0,sizeof(width*height*channels*sizeof(unsigned char)));

	cudaMemcpy(deviceInputImageData, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	dim3 dimGrid(ceil((float) height/TILE_WIDTH), ceil((float) width/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	
	cout <<"PARALLEL" << endl;
	cout << "image dimensions: "<< width1 << "x" << height1 << endl;

	//hole filling
	cout << "Hole Filling elapsed in time: ";
	cudaEventRecord(startEvent);
	par_hole_filling<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height,device_X, device_X1, device_flag);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("para_hole_filling.png", width, height, channels, Parallel, 0);

    return 0;
}
