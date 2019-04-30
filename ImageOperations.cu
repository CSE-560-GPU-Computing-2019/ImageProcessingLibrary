#include "cudaImage.h"

__global__ void paraInverse(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels)
{
	const unsigned char * inputImageData = inputImage;
	
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
	
    for (int k=0; k<channels; k++) {                    //cycle on channels
       
	if(i< dataSizeX && j < dataSizeY)   
	outputImageData[(dataSizeX * i + j)*channels + k] = 255-inputImageData[(dataSizeX * i + j)*channels + k];
			}
       
}

__global__ void paraBrightness(const unsigned char *inputImage, unsigned char *outputImageData, int dataSizeX, int dataSizeY, int channels, int shift)
{
	const unsigned char * inputImageData = inputImage;
	int value;
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
	
    for (int k=0; k<channels; k++) {                    //cycle on channels
       
	if(i< dataSizeX && j < dataSizeY) 
	{
	 value = inputImageData[(dataSizeX * i + j)*channels + k]+shift;
	  if( value >255)
	  {
		value=255;
	  }
	  else if ( value<0)
	  {
		value=0;
	  }
	  outputImageData[(dataSizeX * i + j)*channels + k]=value;
	}
    }   
}

__global__ void parFlipV(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData = inputImage;
    int i =  blockIdx.x* blockDim.x + threadIdx.x ;
    int j =  blockIdx.y* blockDim.y +  threadIdx.y;
    int src_adr=(( i * width ) + (width -1 - j))*channels;
    int dest_adr=(  i * width + j )*channels;	
    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[dest_adr+k] = inputImageData[(src_adr+k)];
    }
}

__global__ void parFlipH(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
	const unsigned char * inputImageData = inputImage;
	
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
	
                  for (int k=0; k<channels; k++) {                    //cycle on channels
               		 outputImageData[( i * width + j)*channels + k] = inputImageData[((height - 1 - i) * width + j)*channels +k];
    }

}

__global__ void parRotateAnti(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
	const unsigned char * inputImageData = inputImage;
	
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	    for (int k=0; k<channels; k++) {                    //cycle on channels
                                         outputImageData[( i * width  + j)*channels + k] = inputImageData[((j * width ) + ( width - 1- i))*channels + k];
	}
}

__global__ void parRotateClock(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
	const unsigned char * inputImageData = inputImage;
	
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
	
    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[( i *width + j) *channels + k] = inputImageData[((height- 1 - j)* width + i)*channels + k];  			}
}

__global__ void parCrop(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels, int x1, int y1, int x2, int y2)
{
    const unsigned char * inputImageData = inputImage;
   int new_h=x2-x1+1;
   int new_w=y2-y1+1;
   int i = threadIdx.x + blockIdx.x* blockDim.x;
   int j = threadIdx.y + blockIdx.y* blockDim.y;
   if (i>new_h || j>new_w) return;	

    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[( i * new_w + j)*channels + k] = inputImageData[((x1+i)*width+(y1+j))*channels+k];
            }
}

__global__ void parNot(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData = inputImage;
    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[( i *width + j)*channels + k] =255 - inputImageData[(i *width + j )*channels + k];
    }
}

__global__ void parAnd(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData1 = inputImage1;
    const unsigned char * inputImageData2 = inputImage2;
    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[( i *width + j)*channels + k] =  inputImageData1[( i *width + j)*channels + k] & inputImageData2[( i *width + j)*channels + k];
              //  outputImageData[( i *width + j)*channels + k] =  inputImageData2[( i *width + j)*channels + k];
    }
}


__global__ void parOr(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData1 = inputImage1;
    const unsigned char * inputImageData2 = inputImage2;
    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[( i *width + j)*channels + k] =  inputImageData1[( i *width + j)*channels + k] | inputImageData2[( i *width + j)*channels + k];
    }
}

__global__ void parXor(const unsigned char *inputImage1, const unsigned char *inputImage2,unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData1 = inputImage1;
    const unsigned char * inputImageData2 = inputImage2;
    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

    for (int k=0; k<channels; k++) {                    //cycle on channels
                outputImageData[( i *width + j)*channels + k] =  inputImageData1[( i *width + j)*channels + k] ^ inputImageData2[( i *width + j)*channels + k];
    }
}

__global__ void par_mean_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   //__shared__ float data[TILE_WIDTH + 2][TILE_WIDTH + 2][channels];
   __shared__ float data[18][18][3];

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	//each thread will copy its own value and value of its four extreme neighbouring cells
	int i1=threadIdx.x;
	int j1=threadIdx.y;
	
	// copying image into shared memory. each thread copies a pixel and its four neighbouring pixels
	for (int k=0;k<channels;k++)
	{
		data[i1+1][j1+1][k]=inputImageData[(i*width+j)*channels+k];
		if (i-1>=0 && j-1>=0)
		{
			data[i1-1+1][j1-1+1][k]=inputImageData[((i-1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1-1+1][k]=0;
		}
		if (i-1>=0 && j+1<width)
		{
			data[i1-1+1][j1+1+1][k]=inputImageData[((i-1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1+1+1][k]=0;
		}
		if (i+1<height && j-1>=0)
		{
			data[i1+1+1][j1-1+1][k]=inputImageData[((i+1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1+1+1][j1-1+1][k]=0;
		}
		if (i+1<height && j+1<width)
		{
			data[i1+1+1][j1+1+1][k]=inputImageData[((i+1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1+2][j1+2][k]=0;
		}
	}
	// waiting for all threads to finish their work
	__syncthreads();
	
	// applying mean filter	
    for (int k=0; k<channels; k++)                     //cycle on channels
    {   
		float temp=0;
		for(int ki=0;ki<=2;ki++)
		{
			for(int kj=0;kj<=2;kj++)
			{
				temp=temp+data[i1+ki][j1+kj][k];
			}
		}
		outputImageData[( i *width + j)*channels + k] = (int) (temp/9.0);
    }
}

__global__ void par_gaussian_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData = inputImage;
    const unsigned char gaus[25]={1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};
    //__shared__ float data[TILE_WIDTH + 4][TILE_WIDTH + 4][channels];
	__shared__ float data[20][20][3];

	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	int i1=threadIdx.x;
	int j1=threadIdx.y;

	// copying image into shared memory
	for (int k=0;k<channels;k++)
	{
	    data[i1+2][j1+2][k]=inputImageData[(i*width+j)*channels+k];
		if (i-2>=0 && j-2>=0)
		{
			data[i1-2+2][j1-2+2][k]=inputImageData[((i-2)*width+j-2)*channels+k];
		}
		else
		{
			data[i1-2+2][j1-2+2][k]=0;
		}
		if (i-2>=0 && j+2<width)
		{
			data[i1-2+2][j1+2+2][k]=inputImageData[((i-2)*width+j+2)*channels+k];
		}
		else
		{
			data[i1-2+2][j1+2+2][k]=0;
		}
		if (i+2<height && j-2>=0)
		{
			data[i1+2+2][j1-2+2][k]=inputImageData[((i+2)*width+j-2)*channels+k];
		}
		else
		{
			data[i1+2+2][j1-2+2][k]=0;
		}
		if (i+2<height && j+2<width)
		{
			data[i1+2+2][j1+2+2][k]=inputImageData[((i+2)*width+j+2)*channels+k];
		}
		else
		{
			data[i1+2+2][j1+2+2][k]=0;
		}
	}
	// waiting for all thread to finish their work
	__syncthreads();
	 	
    for (int k=0; k<channels; k++)                     //cycle on channels
    {       
		float temp=0;
		for (int ki=0;ki<=4;ki++)
		{
			for (int kj=0;kj<=4;kj++)
			{
				temp=temp+data[i1+ki][j1+kj][k]*gaus[ki*5+kj];
			}
		}
		outputImageData[( i *width + j)*channels + k] = (int) (temp/273.0);
    }
}

__global__ void par_median_filter3(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData = inputImage;
    unsigned char median[9]={0,0,0,0,0,0,0,0,0};// 3X3
    __shared__ float data[18][18][3];

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	//each thread will copy its own value and value of its four extreme neighbouring cells
	int i1=threadIdx.x;
	int j1=threadIdx.y;
	for (int k=0;k<channels;k++)
	{
		data[i1+1][j1+1][k]=inputImageData[(i*width+j)*channels+k];
	
		if ((i-1>=0) && (j-1>=0))
		{
			data[i1-1+1][j1-1+1][k]=inputImageData[((i-1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1-1+1][k]=0;
		}
		if ((i-1>=0) && (j+1<width))
		{
			data[i1-1+1][j1+1+1][k]=inputImageData[((i-1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1+1+1][k]=0;
		}
		if ((i+1<height) && (j-1>=0))
		{
			data[i1+1+1][j1-1+1][k]=inputImageData[((i+1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1+1+1][j1-1+1][k]=0;
		}
		if (i+1<height && j+1<width)
		{
			data[i1+1+1][j1+1+1][k]=inputImageData[((i+1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1+2][j1+2][k]=0;
		}
	}
	__syncthreads();
	 
    for (int k=0; k<channels; k++)                     //cycle on channels
    {
		float temp=0;
		int indx=0;
		for (int ki=0;ki<=2;ki++)
		{
			for (int kj=0;kj<=2;kj++)
			{
				//if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) median[indx++]=0;
				median[indx++]=data[i1+ki][j1+kj][k];
			}
		}
		//sorting
        for(int k1=1;k1<9;k1++)
		{
			int temp=median[k1];
			int k2=k1-1;
			while(k2>=0 && median[k2]>temp){
			    median[k2+1]=median[k2];
			    k2--;
			}
			median[k2+1]=temp;
		}
        outputImageData[( i *width + j)*channels + k] = median[4];
    }
}

__global__ void par_median_filter5(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
    const unsigned char * inputImageData = inputImage;
    char median[25];
   __shared__ float data[20][20][3];


	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	int i1=threadIdx.x;
	int j1=threadIdx.y;
	
	
	for (int k=0;k<channels;k++)
	{
	    data[i1+2][j1+2][k]=inputImageData[(i*width+j)*channels+k];
		if (i-2>=0 && j-2>=0)
		{
			data[i1-2+2][j1-2+2][k]=inputImageData[((i-2)*width+j-2)*channels+k];
		}
		else
		{
			data[i1-2+2][j1-2+2][k]=0;
		}
		if (i-2>=0 && j+2<width)
		{
			data[i1-2+2][j1+2+2][k]=inputImageData[((i-2)*width+j+2)*channels+k];
		}
		else
		{
			data[i1-2+2][j1+2+2][k]=0;
		}
		if (i+2<height && j-2>=0)
		{
			data[i1+2+2][j1-2+2][k]=inputImageData[((i+2)*width+j-2)*channels+k];
		}
		else
		{
			data[i1+2+2][j1-2+2][k]=0;
		}
		if (i+2<height && j+2<width)
		{
			data[i1+2+2][j1+2+2][k]=inputImageData[((i+2)*width+j+2)*channels+k];
		}
		else
		{
			data[i1+2+2][j1+2+2][k]=0;
		}
	}
	__syncthreads();
	 
    for (int k=0; k<channels; k++)                     //cycle on channels
    {   
		float temp=0;
		int indx=0;
		for (int ki=0;ki<5;ki++)
		{
			for (int kj=0;kj<5;kj++)
				median[indx++]= data[i1+ki][j1+kj][k];
		}
			    
		//sorting
        for(int k1=1;k1<25;k1++)
		{
			int temp=median[k1];
			int k2=k1-1;
			while(k2>=0 && median[k2]>temp)
			{
			    median[k2+1]=median[k2];
			    k2--;
			}
			median[k2+1]=temp;
		}
        outputImageData[( i *width + j)*channels + k] = median[12];            
    }
}

__global__ void par_max_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   unsigned char max[9]={0,0,0,0,0,0,0,0,0};
   int max_neig;
    __shared__ float data[18][18][3];

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	//each thread will copy its own value and value of its four extreme neighbouring cells
	int i1=threadIdx.x;
	int j1=threadIdx.y;

	for (int k=0;k<channels;k++)
	{
		data[i1+1][j1+1][k]=inputImageData[(i*width+j)*channels+k];
	
		if (i-1>=0 && j-1>=0)
		{
			data[i1-1+1][j1-1+1][k]=inputImageData[((i-1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1-1+1][k]=0;
		}
		if (i-1>=0 && j+1<width)
		{
			data[i1-1+1][j1+1+1][k]=inputImageData[((i-1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1+1+1][k]=0;
		}
		if (i+1<height && j-1>=0)
		{
			data[i1+1+1][j1-1+1][k]=inputImageData[((i+1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1+1+1][j1-1+1][k]=0;
		}
		if (i+1<height && j+1<width)
		{
			data[i1+1+1][j1+1+1][k]=inputImageData[((i+1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1+2][j1+2][k]=0;
		}
	}
	__syncthreads();
	
	for (int k=0; k<channels; k++)                     //cycle on channels
	{	
		float temp=0;
		int indx=0;
		for (int ki=0;ki<=2;ki++)
		{
			for (int kj=0;kj<=2;kj++)
			{
				max[indx++]=data[ki+i1][kj+j1][k];
			}
		}
		
		// finding max from neighbourhood
        max_neig=max[0];
		for(int k1=1;k1<9;k1++)
		{
			if (max[k1]>max_neig) 
				max_neig=max[k1];
		}
        outputImageData[( i *width + j)*channels + k] = max_neig;
    }
}

__global__ void par_min_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
	const unsigned char * inputImageData = inputImage;
	unsigned char min[9]={0,0,0,0,0,0,0,0,0};
	int min_neig;
	__shared__ float data[18][18][3];

	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	//each thread will copy its own value and value of its four extreme neighbouring cells
	int i1=threadIdx.x;
	int j1=threadIdx.y;
	
	for (int k=0;k<channels;k++)
	{
		data[i1+1][j1+1][k]=inputImageData[(i*width+j)*channels+k];
		if (i-1>=0 && j-1>=0)
		{
			data[i1-1+1][j1-1+1][k]=inputImageData[((i-1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1-1+1][k]=255;
		}
		if (i-1>=0 && j+1<width)
		{
			data[i1-1+1][j1+1+1][k]=inputImageData[((i-1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1+1+1][k]=255;
		}
		if (i+1<height && j-1>=0)
		{
			data[i1+1+1][j1-1+1][k]=inputImageData[((i+1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1+1+1][j1-1+1][k]=255;
		}
		if (i+1<height && j+1<width)
		{
			data[i1+1+1][j1+1+1][k]=inputImageData[((i+1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1+2][j1+2][k]=255;
		}
	}
	__syncthreads();
	
    for (int k=0; k<channels; k++)                     //cycle on channels
	{
		float temp=0;
		int indx=0;
		for (int ki=0;ki<=2;ki++)
		{
			for (int kj=0;kj<=2;kj++)
			{
				min[indx++]=data[i1+ki][j1+kj][k];
			}
		}
	
		// finding min from neighbourhood
        min_neig=min[0];
		for(int k1=1;k1<9;k1++)
		{
			if (min[k1]<min_neig) 
				min_neig=min[k1];
		}
        outputImageData[( i *width + j)*channels + k] = min_neig;
    }
}

__global__ void par_edge_detection(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
	const unsigned char * inputImageData = inputImage;
	const char sobelx[9]={-1,0,1,-2,0,2,-1,0,1};
	const char sobely[9]={-1,-2,-1,0,0,0,1,2,1};
	__shared__ float data[18][18][3];

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	//each thread will copy its own value and value of its four extreme neighbouring cells
	int i1=threadIdx.x;
	int j1=threadIdx.y;
	
	for (int k=0;k<channels;k++)
	{
		data[i1+1][j1+1][k]=inputImageData[(i*width+j)*channels+k];
		if (i-1>=0 && j-1>=0)
		{
			data[i1-1+1][j1-1+1][k]=inputImageData[((i-1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1-1+1][k]=0;
		}
		if (i-1>=0 && j+1<width)
		{
			data[i1-1+1][j1+1+1][k]=inputImageData[((i-1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1-1+1][j1+1+1][k]=0;
		}
		if (i+1<height && j-1>=0)
		{
			data[i1+1+1][j1-1+1][k]=inputImageData[((i+1)*width+j-1)*channels+k];
		}
		else
		{
			data[i1+1+1][j1-1+1][k]=0;
		}
		if (i+1<height && j+1<width)
		{
			data[i1+1+1][j1+1+1][k]=inputImageData[((i+1)*width+j+1)*channels+k];
		}
		else
		{
			data[i1+2][j1+2][k]=0;
		}
	}
	__syncthreads();
	
	for (int k=0; k<channels; k++)                     //cycle on channels
	{			
		float tempx=0;
		float tempy=0;
			
		for (int ki=-1;ki<=1;ki++)
		{
			for (int kj=-1;kj<=1;kj++)
			{
				tempx=tempx+data[i1+1+ki][j1+1+kj][k]*sobelx[(ki+1)*3+kj+1];
				tempy=tempy+data[i1+1+ki][j1+1+kj][k]*sobely[(ki+1)*3+kj+1];
			}
		}
			
        outputImageData[( i *width + j)*channels + k] = (int) (sqrt(tempx*tempx+tempy*tempy));
    }
}

__global__ void parchannelSplit(const unsigned char *inputImage1, unsigned char *outputImageData_Red,unsigned char *outputImageData_Green,unsigned char *outputImageData_Blue, int width, int height)
{
    const unsigned char * inputImageData1 = inputImage1;

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
	outputImageData_Red[i *width + j] =  inputImageData1[( i *width + j)*3 ] ;
	outputImageData_Green[ i *width + j ] =  inputImageData1[( i *width + j)*3 + 1] ;
	outputImageData_Blue[ i *width + j] =  inputImageData1[( i *width + j)*3 + 2] ;
}

__global__ void parRGBToGrey(const unsigned char *inputImage1, unsigned char *outputImageData, int width, int height)
{
    const unsigned char * inputImageData1 = inputImage1;
    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

    outputImageData[i *width + j] =  (int) (0.3*inputImageData1[( i *width + j)*3 ]+ 0.59*inputImageData1[( i *width + j)*3 + 1]+0.11*inputImageData1[( i *width + j)*3 + 2]);
}

__global__ void parGreyToBinary(const unsigned char *inputImage1, unsigned char *outputImageData, int width, int height, float level)
{
    const unsigned char * inputImageData1 = inputImage1;
	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

    float temp=inputImageData1[i *width + j]/255.0;
    if (temp<level)	outputImageData[i *width + j] = 0;
    else	outputImageData[i *width + j] = 255;		
}

__global__ void par_adp_local_noise_reduction_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,int channels, int *sum_total, int *sum_square_total)
{
	const unsigned char * inputImageData = inputImage;
	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;

	int k=0;

	//capturing global statistics
	int temp=inputImageData[(i*width+j)*channels+k];
	atomicAdd(sum_total,temp);
    atomicAdd(sum_square_total,temp*temp);
	__syncthreads();
	
	float mean = (*sum_total)/(height*width);
	float var = (*sum_square_total/(height*width))-(mean*mean);
	
	int nh_size=3;
	int rad= (int) ((nh_size-1)/2);
	
	//__shared__ int data[TILE_WIDTH + nh_size-1][TILE_WIDTH + nh_size-1][1];
	__shared__ int data[18][18][1];
	int i1=threadIdx.x;
	int j1=threadIdx.y;

	//copying data into shared memory
	for (int k=0;k<channels;k++)
	{
		data[i1+rad][j1+rad][k]=inputImageData[(i*width+j)*channels+k];
		if (i-rad>=0 && j-rad>=0)
		{
			data[i1][j1][k]=inputImageData[((i-rad)*width+j-rad)*channels+k];
		}
		else
		{
			data[i1][j1][k]=0;
		}
		if (i-rad>=0 && j+rad<width)
		{
			data[i1][j1+rad+rad][k]=inputImageData[((i-rad)*width+j+rad)*channels+k];
		}
		else
		{
			data[i1][j1+rad+rad][k]=0;
		}
		if (i+rad<height && j-rad>=0)
		{
			data[i1+rad+rad][j1][k]=inputImageData[((i+rad)*width+j-rad)*channels+k];
		}
		else
		{
			data[i1+rad+rad][j1][k]=0;
		}
		if (i+rad<height && j+rad<width)
		{
			data[i1+rad+rad][j1+rad+rad][k]=inputImageData[((i+rad)*width+j+rad)*channels+k];
		}
		else
		{
			data[i1+rad+rad][j1+rad+rad][k]=0;
		}
	}
	__syncthreads();
	

	int sum_nh=0;
	int sum_sq_nh=0;
			
	for(int ki=0;ki<nh_size;ki++)
	{
		for(int kj=0;kj<nh_size;kj++)
		{   
		    int temp = data[i1+ki][j1+kj][k];
			sum_nh=sum_nh+temp;
			sum_sq_nh=sum_sq_nh+temp*temp;
		}
	}
	float mean_nh=sum_nh/(nh_size*nh_size);
    float var_nh=sum_sq_nh/(nh_size*nh_size)-(mean_nh*mean_nh);
	float alpha=var/var_nh;
	if (alpha>1.0) 
		alpha=1.0;
	outputImageData[(i*width+j)*channels+k]=(1-alpha)*inputImageData[(i*width+j)*channels+k]+alpha*mean_nh;
}

//Adaptive Median Filter
__global__ void par_adp_median_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,int channels)
{
	const int MaxFilterSize=9;
	const unsigned char * inputImageData = inputImage;
	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
    
	//__shared__ float data[TILE_WIDTH + MaxFilterSize-1][TILE_WIDTH + MaxFilterSize-1][1];
	__shared__ float data[24][24][1];
	
	int i1=threadIdx.x;
	int j1=threadIdx.y;

	int filter_size=3;
	int rad=(int)((MaxFilterSize-1)/2);
	for (int k=0;k<channels;k++)
	{
		data[i1+rad][j1+rad][k]=inputImageData[(i*width+j)*channels+k];
		if (i-rad>=0 && j-rad>=0)
		{
			data[i1][j1][k]=inputImageData[((i-rad)*width+j-rad)*channels+k];
		}
		else
		{
			data[i1][j1][k]=0;
		}
		if (i-rad>=0 && j+rad<width)
		{
			data[i1][j1+rad+rad][k]=inputImageData[((i-rad)*width+j+rad)*channels+k];
		}
		else
		{
			data[i1][j1+rad+rad][k]=0;
		}
		if (i+rad<height && j-rad>=0)
		{
			data[i1+rad+rad][j1][k]=inputImageData[((i+rad)*width+j-rad)*channels+k];
		}
		else
		{
			data[i1+rad+rad][j1][k]=0;
		}
		if (i+rad<height && j+rad<width)
		{
			data[i1+rad+rad][j1+rad+rad][k]=inputImageData[((i+rad)*width+j+rad)*channels+k];
		}
		else
		{
			data[i1+rad+rad][j1+rad+rad][k]=0;
		}
	}
	__syncthreads();
	
    int flag=0;
	while(!flag)
	{
		unsigned char *neighbour= new unsigned char[filter_size*filter_size];
		int rad1=(int)((filter_size-1)/2);
		int c1=0;
		int k = 0;
		for (int ki=-rad1;ki<=rad1;ki++)
		{
			for (int kj=-rad1;kj<=rad1;kj++)
			{
			    neighbour[c1++]=data[i1+rad+ki][j1+rad+kj][k];
			}
		}

		//sort(neighbour,filter_size*filter_size);
		int size =filter_size*filter_size;
		for(int i=1;i<size;i++)
		{
			unsigned char temp=neighbour[i];
			int j=i-1;
			while(j>=0 && neighbour[j]>temp)
			{
				neighbour[j+1]=neighbour[j];
				j=j-1;
			}
			neighbour[j+1]=temp;
		}
	
		unsigned char z_min=neighbour[0];
		unsigned char z_max=neighbour[filter_size*filter_size-1];
		unsigned char z_med=neighbour[(filter_size*filter_size-1)/2];
		int a1=z_med - z_min;
		int a2=z_med - z_max;
		if (a1>0 && a2<0)
		{
			unsigned char z_xy=inputImage[(i*width+j)*channels+k];
			int b1 = z_xy - z_min;
			int b2 = z_xy - z_max;
			if (b1>0 && b2<0 ) 
				outputImageData[(i*width+j)*channels+k]=z_xy;
			else 
				outputImageData[(i*width+j)*channels+k]=z_med;
			flag=1;  
		}
		else
		{
			filter_size=filter_size+2;
			if (filter_size> MaxFilterSize) 
			{ 
				outputImageData[(i*width+j)*channels+k]=z_med;
				flag=1;
			}
		}
	}
}

__global__ void parDialation_binary(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
	int rad=1;
	//__shared__ unsigned char data[TILE_WIDTH + 2][TILE_WIDTH + 2];
	__shared__ unsigned char data[18][18];

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	int i1=threadIdx.x;
	int j1=threadIdx.y;
    
	//copying data into shared memory
	data[i1+rad][j1+rad]=inputImageData[i*width+j];
	if (i-rad>=0 && j-rad>=0)
	{
			data[i1][j1]=inputImageData[((i-rad)*width+j-rad)];
	}
	else
	{
		data[i1][j1]=0;
	}
	if (i-rad>=0 && j+rad<width)
	{
		data[i1][j1+rad+rad]=inputImageData[((i-rad)*width+j+rad)];
	}
	else
	{
		data[i1][j1+rad+rad]=0;
	}
	if (i+rad<height && j-rad>=0)
	{
		data[i1+rad+rad][j1]=inputImageData[((i+rad)*width+j-rad)];
	}
	else
	{
		data[i1+rad+rad][j1]=0;
	}
	if (i+rad<height && j+rad<width)
	{
		data[i1+rad+rad][j1+rad+rad]=inputImageData[((i+rad)*width+j+rad)];
	}
	else
	{
		data[i1+rad+rad][j1+rad+rad]=0;
	}
	__syncthreads();
	int flag=0;
        for(int ki=-rad; ki<=rad && flag==0; ki++)
	{
	    for(int kj=-rad; kj<=rad; kj++)
	    {
			if ( B[rad+ki][rad+kj]==1 && (data[i1+rad+ki][j1+rad+kj]==255)) 
			{		
				flag=1;
				break;
			}
		}
	}
	if (flag==1)
	    outputImageData[i *width + j] =  255;
        else
		outputImageData[i *width + j] =  0;

}

__global__ void parErode_binary(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
        const unsigned char * inputImageData= inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
        int i = threadIdx.x + blockIdx.x* blockDim.x;
        int j = threadIdx.y + blockIdx.y* blockDim.y;
	//__shared__ unsigned char data[TILE_WIDTH + 2][TILE_WIDTH + 2];
	__shared__ unsigned char data[18][18];
	
    int i1=threadIdx.x;
	int j1=threadIdx.y;
   
	int rad=1;

	data[i1+rad][j1+rad]=inputImageData[i*width+j];
	if (i-rad>=0 && j-rad>=0)
	{
			data[i1][j1]=inputImageData[((i-rad)*width+j-rad)];
	}
	else
	{
		data[i1][j1]=0;
	}
	if (i-rad>=0 && j+rad<width)
	{
		data[i1][j1+rad+rad]=inputImageData[((i-rad)*width+j+rad)];
	}
	else
	{
		data[i1][j1+rad+rad]=0;
	}
	if (i+rad<height && j-rad>=0)
	{
		data[i1+rad+rad][j1]=inputImageData[((i+rad)*width+j-rad)];
	}
	else
	{
		data[i1+rad+rad][j1]=0;
	}
	if (i+rad<height && j+rad<width)
	{
		data[i1+rad+rad][j1+rad+rad]=inputImageData[((i+rad)*width+j+rad)];
	}
	else
	{
		data[i1+rad+rad][j1+rad+rad]=0;
	}
	__syncthreads();
	
        int flag=1;
	
	for(int ki=-rad; ki<=rad && flag==1; ki++)
	{
	    for(int kj=-rad;kj<=rad;kj++)
	    {
			if ( B[rad+ki][rad+kj]==1 && (data[i1+rad+ki][j1+rad+kj]!=255)) 
			{		
				flag=0;
				break;
			}
		}
	}
	if (flag==1)
	    outputImageData[i *width + j] =  255;
    else
		outputImageData[i *width + j] =  0;

}

__global__ void parDialation_Grey(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
	int rad=1;
	//__shared__ unsigned char data[TILE_WIDTH + 2][TILE_WIDTH + 2];
	__shared__ unsigned char data[18][18];
	
	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
    int i1=threadIdx.x;
	int j1=threadIdx.y;
    
	data[i1+rad][j1+rad]=inputImageData[i*width+j];
	if (i-rad>=0 && j-rad>=0)
	{
			data[i1][j1]=inputImageData[((i-rad)*width+j-rad)];
	}
	else
	{
		data[i1][j1]=0;
	}
	if (i-rad>=0 && j+rad<width)
	{
		data[i1][j1+rad+rad]=inputImageData[((i-rad)*width+j+rad)];
	}
	else
	{
		data[i1][j1+rad+rad]=0;
	}
	if (i+rad<height && j-rad>=0)
	{
		data[i1+rad+rad][j1]=inputImageData[((i+rad)*width+j-rad)];
	}
	else
	{
		data[i1+rad+rad][j1]=0;
	}
	if (i+rad<height && j+rad<width)
	{
		data[i1+rad+rad][j1+rad+rad]=inputImageData[((i+rad)*width+j+rad)];
	}
	else
	{
		data[i1+rad+rad][j1+rad+rad]=0;
	}
	__syncthreads();
	
    
	int max=0;
	for(int ki=-rad;ki<=rad;ki++)
	{
	    for(int kj=-rad;kj<=rad;kj++)
	    {
			if ( B[rad+ki][rad+kj]==1 && (data[i1+rad+ki][j1+rad+kj]>max)) 
			{		
				max=data[i1+rad+ki][j1+rad+kj];
			}
		}
	}
	outputImageData[i *width + j] =  max;
}

__global__ void parErode_Grey(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    const unsigned char * inputImageData = inputImage;
	unsigned int B[3][3]={1,1,1,1,1,1,1,1,1};
    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	int flag=1;
//	__shared__ unsigned char data[TILE_WIDTH + 2][TILE_WIDTH + 2];
	__shared__ unsigned char data[18][18];
	int rad=1;
	
    int i1=threadIdx.x;
	int j1=threadIdx.y;
    
	data[i1+rad][j1+rad]=inputImageData[i*width+j];
	if (i-rad>=0 && j-rad>=0)
	{
			data[i1][j1]=inputImageData[((i-rad)*width+j-rad)];
	}
	else
	{
		data[i1][j1]=255;
	}
	if (i-rad>=0 && j+rad<width)
	{
		data[i1][j1+rad+rad]=inputImageData[((i-rad)*width+j+rad)];
	}
	else
	{
		data[i1][j1+rad+rad]=255;
	}
	if (i+rad<height && j-rad>=0)
	{
		data[i1+rad+rad][j1]=inputImageData[((i+rad)*width+j-rad)];
	}
	else
	{
		data[i1+rad+rad][j1]=255;
	}
	if (i+rad<height && j+rad<width)
	{
		data[i1+rad+rad][j1+rad+rad]=inputImageData[((i+rad)*width+j+rad)];
	}
	else
	{
		data[i1+rad+rad][j1+rad+rad]=255;
	}
	__syncthreads();
	
    
	int min=255;
	for(int ki=-rad;ki<=rad;ki++)
	{
	    for(int kj=-rad;kj<=rad;kj++)
	    {
			if ( B[rad+ki][rad+kj]==1 && (data[i1+rad+ki][j1+rad+kj]<min)) 
			{		
				min=data[i1+rad+ki][j1+rad+kj];
			}
		}
	}
	outputImageData[i *width + j] =  min;
}


__global__ void computeFreq(unsigned char*inputImg,float*freq, int size,int channels)
{
	__shared__ unsigned int shared_freq[256];

	// initialize shared memory
	if(threadIdx.x < 256) shared_freq[threadIdx.x]=0;
	
	__syncthreads();
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	
	if(i< size)
	{
		atomicAdd(&shared_freq[inputImg[i]],1);
	}
	
	__syncthreads();
	
	if(threadIdx.x < 256) // adding up freq from all the shared memory blocks
	{
		atomicAdd(&freq[threadIdx.x],shared_freq[threadIdx.x]);
	}
}

__global__ void computeCDF(float*cdf,float *par_freq, int size)
{
	__shared__  float sharedCDF[256];
        int i=threadIdx.x+blockDim.x*blockIdx.x;

	if(threadIdx.x<256)
	{
	//	printf("%d %d:\n",threadIdx.x,blockIdx.x);
		sharedCDF[threadIdx.x]=par_freq[threadIdx.x]/size;
	}
	
	__syncthreads();
	
	// Reduction step
	int stride=1,index;
	while(stride<256)
	{
		 index =(threadIdx.x+1)*stride*2 -1; // index to write updated element
		if(index <256)
		  sharedCDF[index]+=sharedCDF[index-stride];
		stride=stride*2;
		
		__syncthreads();
	}
	
	//post reduction step
	stride =256/4;
	while(stride>0)
	{
	 index =(threadIdx.x+1)*stride*2 -1; 
	 if(index+stride<256  )
		sharedCDF[index+stride]+=sharedCDF[index];
		
	stride=stride/2;
	__syncthreads();
	}
	
	 //int i=threadIdx.x+blockDim.x*blockIdx.x;
	 if(threadIdx.x <256 )
	 {
		cdf[i]+=sharedCDF[threadIdx.x];
	//	atomicAdd(&cdf[threadIdx.x],sharedCDF[threadIdx.x]);
	 }
	
}

__global__ void computeHistogramEqualization(const unsigned char*inputImg, unsigned char*OutputImg,float*cdf,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	OutputImg[i]=255*cdf[inputImg[i]];
}
 
