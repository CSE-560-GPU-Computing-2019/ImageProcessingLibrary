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

/*__global__ void parFlipV(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
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

/*__global__ void par_gaussian_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   const unsigned char gaus[25]={1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};
    
	int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
    for (int k=0; k<channels; k++) {                    //cycle on channels
          float temp=0;
			
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

__global__ void par_mean_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;

    int i = threadIdx.x + blockIdx.x* blockDim.x;
    int j = threadIdx.y + blockIdx.y* blockDim.y;
	
    for (int k=0; k<channels; k++) {                    //cycle on channels
                float temp=0;
			
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
 } */

