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
#define nh_size 3
#define MaxFilterSize 9

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

void seqContrast(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels,float* freq,float* cumm_freq)
{
   const unsigned char * inputImageData = inputImage;

	unsigned char *new_grey_level= new unsigned char[channels*256];


   int temp,curr;
    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
	
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   
			      
				temp=(int) inputImageData[(i*width+j)*channels+k];
				
				 freq[temp]++;
            }
        }
		
		for (int i=0;i<256;i++){freq[i]=freq[i]/(width*height);}
	cumm_freq[0]=freq[0];
	for (int i=1;i<256;i++) cumm_freq[i]=cumm_freq[i-1]+freq[i];
	for (int i=0;i<256;i++) new_grey_level[k*256+i]=(255*cumm_freq[i]);
    }
     	
	 temp=0;
	double value;
	                   //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {       value=0.0;
		    for (int k=0; k<channels; k++) { 
	            temp=inputImageData[(i*width+j)*channels+k];
	            value+=new_grey_level[k*256+temp];
		   }
	
		    if(channels>1)
		    {
		     for (int k=0; k<channels; k++) {
			 outputImageData[(i*width+j)*channels+k]=value/3;
	               }
		    }
		    else
			outputImageData[(i*width+j)]=value;

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

void seq_median_filter3(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   unsigned char median[9]={0,0,0,0,0,0,0,0,0};

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float temp=0;
			    int indx=0;
			    for (int i1=-1;i1<=1;i1++)
				{
					for (int j1=-1;j1<=1;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) median[indx++]=0;
						 else median[indx++]=inputImageData[(((i+i1)*width+j+j1)*channels)+k];
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
    }
}

void seq_median_filter5(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{   
   const unsigned char * inputImageData = inputImage;
   char median[25];

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float temp=0;
			    int indx=0;
			    for (int i1=-2;i1<=2;i1++)
				{
					for (int j1=-2;j1<=2;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) median[indx++]=0;
						 else median[indx++]=inputImageData[(((i+i1)*width+j+j1)*channels)+k];
					}
				}
			    //sorting
                for(int k1=1;k1<25;k1++)
				{
					int temp=median[k1];
					int k2=k1-1;
					while(k2>=0 && median[k2]>temp){
					       median[k2+1]=median[k2];
						   k2--;
					}
					median[k2+1]=temp;
				}
                outputImageData[( i *width + j)*channels + k] = median[12];
            }
        }
    }
}

void seq_max_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   unsigned char max[9]={0,0,0,0,0,0,0,0,0};
   int max_neig;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float temp=0;
			    int indx=0;
			    for (int i1=-1;i1<=1;i1++)
				{
					for (int j1=-1;j1<=1;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) max[indx++]=0;
						 else max[indx++]=inputImageData[(((i+i1)*width+j+j1)*channels)+k];
					}
				}
			    // finding max from neighbourhood
                max_neig=max[0];
				for(int k1=1;k1<9;k1++)
				{
					if (max[k1]>max_neig) max_neig=max[k1];
				}
                outputImageData[( i *width + j)*channels + k] = max_neig;
            }
        }
    }
}

void seq_min_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   unsigned char min[9]={0,0,0,0,0,0,0,0,0};
   int min_neig;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float temp=0;
			    int indx=0;
			    for (int i1=-1;i1<=1;i1++)
				{
					for (int j1=-1;j1<=1;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) min[indx++]=0;
						 else min[indx++]=inputImageData[(((i+i1)*width+j+j1)*channels)+k];
					}
				}
			    // finding max from neighbourhood
                min_neig=min[0];
				for(int k1=1;k1<9;k1++)
				{
					if (min[k1]<min_neig) min_neig=min[k1];
				}
                outputImageData[( i *width + j)*channels + k] = min_neig;
            }
        }
    }
}

void seq_edge_detection(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height, int channels)
{
   const unsigned char * inputImageData = inputImage;
   const char sobelx[9]={-1,0,1,-2,0,2,-1,0,1};
   const char sobely[9]={-1,-2,-1,0,0,0,1,2,1};

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (int i = 0; i < height; ++i)                //cycle on image rows
        {
            for (int j = 0; j < width ; ++j)            //cycle on image columns
            {   float tempx=0;
			    float tempy=0;
			
			    for (int i1=-1;i1<=1;i1++)
				{
					for (int j1=-1;j1<=1;j1++)
					{
					     if ((i+i1)<0 || (j+j1)<0 || (i+i1)>=height || (j+j1)>=width) continue;
						 tempx=tempx+inputImageData[(((i+i1)*width+j+j1)*channels)+k]*sobelx[(i1+1)*3+j1+1];
						 tempy=tempy+inputImageData[(((i+i1)*width+j+j1)*channels)+k]*sobely[(i1+1)*3+j1+1];
					}
				}
			
                outputImageData[( i *width + j)*channels + k] = (int) (sqrt(tempx*tempx+tempy*tempy));
            }
        }
    }
}

void seqchannelSplit(const unsigned char *inputImage1, unsigned char *outputImageData_Red,unsigned char *outputImageData_Green,unsigned char *outputImageData_Blue, int width, int height)
{
    int i, j;
    const unsigned char * inputImageData1 = inputImage1;

	for (i = 0; i < height; ++i)                //cycle on image rows
    {
            for (j = 0; j < width ; ++j)            //cycle on image columns
            {
                outputImageData_Red[ i *width + j] =  inputImageData1[( i *width + j)*3 ] ;
				outputImageData_Green[ i *width + j] =  inputImageData1[( i *width + j)*3 + 1] ;
				outputImageData_Blue[ i *width + j] =  inputImageData1[( i *width + j)*3 + 2] ;
            }
    }
}

void seqRGBToGrey(const unsigned char *inputImage1, unsigned char *outputImageData, int width, int height)
{
    int i, j;
    const unsigned char * inputImageData1 = inputImage1;

	for (i = 0; i < height; ++i)                //cycle on image rows
    {
        for (j = 0; j < width ; ++j)            //cycle on image columns
        {
            outputImageData[i *width + j] =  (int) (0.3*inputImageData1[( i *width + j)*3 ]+ 0.59*inputImageData1[( i *width + j)*3 + 1]+0.11*inputImageData1[( i *width + j)*3 + 2]);
	    }
    }
}

void seqGreyToBinary(const unsigned char *inputImage1, unsigned char *outputImageData, int width, int height, float level)
{
    int i, j;
    const unsigned char * inputImageData1 = inputImage1;

	for (i = 0; i < height; ++i)                //cycle on image rows
    {
        for (j = 0; j < width ; ++j)            //cycle on image columns
        {
		    float temp=inputImageData1[i *width + j]/255.0;
            if (temp<level)	outputImageData[i *width + j] = 0;
            else	outputImageData[i *width + j] = 255;		
		}
    }
}

void adp_local_noise_reduction_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,int channels)
{
	const unsigned char * inputImageData = inputImage;
	int sum_total=0;
	int sum_square_total=0;
	//int nh_size=3;//neighbourhood size 3X3
	int rad=floor(nh_size/2);
	//capturing global statistics
	int k=0;
	float mean;
	float var;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			sum_total=sum_total+inputImageData[(i*width+j)*channels+k];
			sum_square_total=sum_square_total+inputImageData[(i*width+j)*channels+k]*inputImageData[(i*width+j)*channels+k];
		}
	}
	mean=sum_total/(height*width);
	var=(sum_square_total/(height*width))-(mean*mean);
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
		    int sum_nh=0;
			int sum_sq_nh=0;
			
			for(int ki=-rad;ki<=rad;ki++)
			{
				for(int kj=-rad;kj<=rad;kj++)
				{   
				    if ((i+ki)<0 || (i+ki)>=height || (j+kj)<0 || (j+kj)>=width) continue;
					int temp=inputImageData[((i+ki)*width+(j+kj))*channels+k];
					sum_nh=sum_nh+temp;
					sum_sq_nh=sum_sq_nh+temp*temp;
				}
			}
			float mean_nh=sum_nh/(nh_size*nh_size);
            float var_nh=sum_sq_nh/(nh_size*nh_size)-(mean_nh*mean_nh);
			float alpha=var/var_nh;
			if (alpha>1.0) alpha=1.0;
			outputImageData[(i*width+j)*channels+k]=(1-alpha)*inputImageData[(i*width+j)*channels+k]+alpha*mean_nh;
		}
	}
}

void sort(unsigned char *a,int size)
{
	for(int i=1;i<size;i++)
	{
		unsigned char temp=a[i];
		int j=i-1;
		while(j>=0 && a[j]>temp)
		{
			a[j+1]=a[j];
			j=j-1;
		}
		a[j+1]=temp;
	}
}

void adp_median_filter(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height,int channels)
{
	//const int MaxFilterSize=7;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
		  int filter_size=3;
		  int flag=0;
		  while(!flag)
		  {
			unsigned char *neighbour= new unsigned char[filter_size*filter_size];
			int rad=floor(filter_size/2);
			int c1=0;
			int k=0;
			for (int ki=-1*rad;ki<=rad;ki++)
			{
				for (int kj=-1*rad;kj<=rad;kj++)
				{
				    if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width)
						neighbour[c1++]=0;
					else
						neighbour[c1++]=inputImage[((i+ki)*width+j+kj)*channels+k];
				}
			}
			sort(neighbour,filter_size*filter_size);
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
				if (b1>0 && b2<0 ) outputImageData[(i*width+j)*channels+k]=z_xy;
				else outputImageData[(i*width+j)*channels+k]=z_med;
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
	}
}

void seqDialation_binary(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
    int rad=1;
	for (i = 0; i < height; ++i)                //cycle on image rows
    {
        for (j = 0; j < width ; ++j)            //cycle on image columns
        {  
		    int flag=0;
                   // int t1=inputImage[i*width+j];
		   // cout<<t1<<" ";
		    for(int ki=-rad; ki<=rad && flag==0; ki++)
			{
			    for(int kj=-rad;kj<=rad;kj++)
			    {
					if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width) continue;
					if ( B[rad+ki][rad+kj]==1 && inputImageData[((i+ki)*width + j+kj)]==255) 
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
    }
}

void seqErode_binary(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
	int rad =1;

	for (i = 0; i < height; ++i)                //cycle on image rows
    {
        for (j = 0; j < width ; ++j)            //cycle on image columns
        {  
		    int flag=1;
		    for(int ki=-rad; ki<=rad && flag==1; ki++)
			{
			    for(int kj=-rad; kj<=rad; kj++)
			    {
					if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width) continue;
					if ( B[rad+ki][rad+kj]==1 && inputImageData[((i+ki)*width + j+kj)]!=255) 
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
    }
}

void seqDialation_Grey(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
    int rad=1;
	for (i = 0; i < height; ++i)                //cycle on image rows
    {
        for (j = 0; j < width ; ++j)            //cycle on image columns
        {  
			int max=0;
		    for(int ki=-rad;ki<=rad;ki++)
			{
			    for(int kj=-rad;kj<=rad;kj++)
			    {
					if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width) continue;
					if ( B[rad+ki][rad+kj]==1 && inputImageData[((i+ki)*width + j+kj)]>max) 
					{		
						max=inputImageData[( i+ki)*width + j+kj];
					}
				}
			}
			outputImageData[i *width + j] =  max;
		}
    }
}

void seqErode_Grey(const unsigned char *inputImage, unsigned char *outputImageData, int width, int height)
{
    int i, j;
    const unsigned char * inputImageData = inputImage;
	unsigned char B[3][3]={1,1,1,1,1,1,1,1,1};
	int rad=1;

	for (i = 0; i < height; ++i)                //cycle on image rows
    {
        for (j = 0; j < width ; ++j)            //cycle on image columns
        {  
		    int min=255;
		    for(int ki=-rad;ki<=rad;ki++)
			{
			    for(int kj=-rad;kj<=rad;kj++)
			    {
					if (i+ki<0 || i+ki>=height || j+kj<0 || j+kj>=width) continue;
					if ( B[rad+ki][rad+kj]==1 && inputImageData[( i+ki)*width + j+kj]<min) 
					{		
						min=inputImageData[( i+ki)*width + j+kj];
					}
				}
			}
			outputImageData[i *width + j] =  min;
		}
    }
}

float paraContrast(const unsigned char*inputImg, unsigned char*OutputImg,int width,int height, int channels, float *par_freq,float *par_cdf)
 {
	unsigned char *deviceInputImageData;
    unsigned char *deviceOutputImageData;
	 float* freq;
	 float *cdf;
  
	cudaEvent_t startEvent,stopEvent;
	cudaDeviceReset();
    cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	int size=height*width;
		
	cudaMalloc((void **) &deviceInputImageData, width * height *channels * sizeof(unsigned int));
	cudaMalloc((void **) &deviceOutputImageData, width * height *channels * sizeof(unsigned int));
	cudaMemcpy(deviceInputImageData, inputImg, width * height * channels * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	
	clock_t begin_time = clock();	
       
	
	
	cudaMalloc((void **) &freq, width * height *channels * sizeof( float));
	cudaMemset(freq,0,256*sizeof(float));
	
	cudaMalloc((void **) &cdf, width * height *channels * sizeof(float));
	cudaMemset(cdf,0,256*sizeof(float));
	
	
	dim3 dimGrid(ceil((width*height*channels)/256));
	dim3 dimGridCDF(1,1,1);
	dim3 dimBlock(256,1,1);
	dim3 dimBlockHE(16,16,1);
	dim3 dimGridHE(ceil(width*height*channels)/256,ceil(width*height*channels)/256);	
	//cout << "\nContrast elapsed in time: ";
	float ms;
	cudaEventRecord(startEvent);
//	computeFreq<<<dimGrid,dimBlock>>>(deviceInputImageData, freq, size, channels);
//	computeCDF<<<dimGridCDF,dimBlock>>>(cdf,freq,size);
//	computeHistogramEqualization<<<dimGrid,dimBlock>>>(deviceInputImageData,deviceOutputImageData,cdf,width);
	
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&ms,startEvent, stopEvent);

	//cout<< ms <<" ms"<<endl;
	cudaMemcpy(par_freq, freq, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(par_cdf, cdf, 256 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(OutputImg, deviceOutputImageData, width*height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//	cout<<"check point : paraContrast";
	return ms;
 }
 
int main()
{
	int channels, channelRGB, channelGS;
	
    // Set Channel Value;
	channels = 3;
	channelRGB = 3;
	channelGS = 1;
	
	int width, height, width1, height1, widthRGB, heightRGB, widthGS, heightGS, widthBin, heightBin, bpp, bpp1, bppRGB, bppGS, bppBin;
   	unsigned char *sequential, *sequentialGS, *seq_img_crop, *sequentialBin;
	const unsigned char* image, *image1, *imageRGB, *imageGS, *imageBin;
	float runTime;
			
	image = stbi_load( "InputRGB/lena256.png", &width, &height, &bpp, channels );
    image1 = stbi_load( "InputRGB/lenaFlip256.png", &width1, &height1, &bpp1, channels );
	imageRGB = stbi_load( "InputRGB/lena256.png", &widthRGB, &heightRGB, &bppRGB, channelRGB );
	imageGS = stbi_load( "InputGS/lena256.png", &widthGS, &heightGS, &bppGS, channelGS );
	imageBin = stbi_load( "InputBin/lena256.png", &widthBin, &heightBin, &bppBin, channelGS );
    
	sequential = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));
	sequentialGS = (unsigned char*)malloc(widthGS*heightGS*channelGS*sizeof(unsigned char));
	sequentialBin = (unsigned char*)malloc(widthBin*heightBin*channelGS*sizeof(unsigned char));
	
	unsigned char *seq_img_red =(unsigned char*)malloc(widthRGB*heightRGB*sizeof(unsigned char));
	unsigned char *seq_img_green =(unsigned char*)malloc(widthRGB*heightRGB*sizeof(unsigned char));
	unsigned char *seq_img_blue =(unsigned char*)malloc(widthRGB*heightRGB*sizeof(unsigned char));
	memset(seq_img_red,0,sizeof(widthRGB*heightRGB*sizeof(unsigned char)));
    memset(seq_img_green,0,sizeof(widthRGB*heightRGB*sizeof(unsigned char)));
    memset(seq_img_blue,0,sizeof(widthRGB*heightRGB*sizeof(unsigned char)));
    
	unsigned char *seq_grey =(unsigned char*)malloc(widthRGB*heightRGB*sizeof(unsigned char));
	
	unsigned char *seq_bin=(unsigned char*)malloc(widthGS*heightGS*sizeof(unsigned char));
	
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
	
	//Contrast
	float frequency[256]={0};
	float cdf[256]={0};
	cout << "CONTRAST elapsed in time: ";
    begin_time = clock();
	seqContrast(image, sequential, width, height, channels,frequency,cdf);
	runTime = (float)( clock() - begin_time ) / ( CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("Contrast.png", width, height, channels, sequential, 0);
	
	
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
	
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	// Mean Filter
	cout << "Mean Filter elapsed in time: ";
    begin_time = clock();   
	seq_mean_filter(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("MeanFilter.png", width, height, channels, sequential, 0);
		
	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	// Gaussian Filter
	cout << "Gaussian Filter elapsed in time: ";
    begin_time = clock();   
	seq_gaussian_filter(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("GaussianFilter.png", width, height, channels, sequential, 0);

	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	// Edge Detection
	cout << "Edge Detection elapsed in time: ";
    begin_time = clock();   
	seq_edge_detection(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("EdgeDetection.png", width, height, channels, sequential, 0);

	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	// Median3 Filter
	cout << "Median3 Filter elapsed in time: ";
    begin_time = clock();   
	seq_median_filter3(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("Median3Filter.png", width, height, channels, sequential, 0);

	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));

	// Median5 Filter
	cout << "Median5 Filter elapsed in time: ";
    begin_time = clock();   
	seq_median_filter5(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("Median5Filter.png", width, height, channels, sequential, 0);

	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	// Max Filter
	cout << "Max Filter elapsed in time: ";
    begin_time = clock();   
	seq_max_filter(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("MaxFilter.png", width, height, channels, sequential, 0);

	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	// Min Filter
	cout << "Min Filter elapsed in time: ";
    begin_time = clock();   
	seq_min_filter(image, sequential, width, height, channels);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("MinFilter.png", width, height, channels, sequential, 0);

	memset(sequential,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Channel Split
	cout << "Channel Split elapsed in time: ";
    begin_time = clock();   
	seqchannelSplit(imageRGB, seq_img_red, seq_img_green, seq_img_blue, widthRGB, heightRGB);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("seq_channelRED.png", widthRGB, heightRGB, 1, seq_img_red, 0);
	stbi_write_png("seq_channelGreen.png", widthRGB, heightRGB, 1, seq_img_green, 0);
	stbi_write_png("seq_channelBlue.png", widthRGB, heightRGB, 1, seq_img_blue, 0);

	memset(seq_img_red,0,sizeof(widthRGB*heightRGB*channelRGB*sizeof(unsigned char)));
	memset(seq_img_blue,0,sizeof(widthRGB*heightRGB*channelRGB*sizeof(unsigned char)));
	memset(seq_img_green,0,sizeof(widthRGB*heightRGB*channelRGB*sizeof(unsigned char)));

	//RGB to GrayScale
	cout << "RGB to GrayScale elapsed in time: ";
    begin_time = clock();   
	seqRGBToGrey(imageRGB, seq_grey, widthRGB, heightRGB); 
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("seq_grey.png", widthRGB, heightRGB, 1, seq_grey, 0);
	
	memset(seq_grey,0,sizeof(widthRGB*heightRGB*channelRGB*sizeof(unsigned char)));
	
	//GrayScale to Binary
	cout << "GrayScale to Binary elapsed in time: ";
    begin_time = clock();   
	seqGreyToBinary(imageGS, seq_bin, widthGS, heightGS, 0.4);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("seq_binary.png", widthGS, heightGS, 1, seq_bin, 0);

	memset(seq_bin,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));

	//Adaptive Noise Reduction Filter
	cout << "Adaptive Noise Reduction Filter elapsed in time: ";
    begin_time = clock();   
	adp_local_noise_reduction_filter(imageGS, sequentialGS, widthGS, heightGS, channelGS);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("ADP_noise.png", widthGS, heightGS, channelGS, sequentialGS, 0);
	
	memset(sequentialGS,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));

	//Adaptive Median Filter
	cout << "Adaptive Median Filter elapsed in time: ";
    begin_time = clock();   
	adp_median_filter(imageGS, sequentialGS, widthGS, heightGS, channelGS);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<< runTime <<" ms" <<endl;
	stbi_write_png("ADP_median.png", widthGS, heightGS, channelGS, sequentialGS, 0);

	// Dialation binary
	cout << "Dialation Binary elapsed in time: ";
    begin_time = clock();   
	seqDialation_binary(imageBin, sequentialBin, widthBin, heightBin);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<<runTime <<" ms" <<endl;
	stbi_write_png("seqDialation_binary.png", widthBin, heightBin, channelGS, sequentialBin, 0);
		
	memset(sequentialBin,0,sizeof(widthBin*heightBin*channelGS*sizeof(unsigned char)));
	
	// Erode binary
	cout << "Erode Binary elapsed in time: ";
    begin_time = clock();   
	seqErode_binary(imageBin, sequentialBin, widthBin, heightBin);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<<runTime <<" ms" <<endl;
	stbi_write_png("seqErode_binary.png", widthBin, heightBin, channelGS, sequentialBin, 0);
	
	memset(sequentialBin,0,sizeof(widthBin*heightBin*channelGS*sizeof(unsigned char)));
	
	// Dialation Grey
	cout << "Dialation Grey elapsed in time: ";
    begin_time = clock();   
	seqDialation_Grey(imageGS, sequentialGS, widthGS, heightGS);
	runTime = (float)( clock() - begin_time ) /  (CLOCKS_PER_SEC/1000);
	cout<<runTime <<" ms" <<endl;
	stbi_write_png("seqDialation_Grey.png", widthGS, heightGS, channelGS, sequentialGS, 0);

	memset(sequentialGS,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));
	
	// Erode Grey
	cout << "Erode Grey elapsed in time: ";
    begin_time = clock();   
	seqErode_Grey(imageGS, sequentialGS, widthGS, heightGS);
	runTime = (float)( clock() - begin_time ) / (CLOCKS_PER_SEC/1000);
	cout<<runTime <<" ms" <<endl;
	stbi_write_png("seqErode_Grey.png", widthGS, heightGS, channelGS, sequentialGS, 0);

	memset(sequentialGS,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));

	cout << "*----------------------------------*" << endl;

/******************************************************************************************************************/	
	
	cudaEvent_t startEvent,stopEvent;
		
	unsigned char *deviceInputImageData,*deviceInputImageData1, *deviceInputImageDataRGB, *deviceInputImageDataGS, *deviceInputImageDataBin;
    unsigned char *deviceOutputImageData, *deviceOutputImageData1, *deviceOutputImageDataR, *deviceOutputImageDataG, *deviceOutputImageDataB, *deviceOutputImageDataGS, *deviceOutputImageDataBin, *deviceOutputImageDataGrS, *deviceOutputImageDataBini;
    runTime=0.0;
	cudaDeviceReset();
    cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);	
	unsigned char *Parallel = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));
	unsigned char *Para_crop;
	unsigned char *ParallelR = (unsigned char*)malloc(widthRGB*heightRGB*channelRGB*sizeof(unsigned char));
	unsigned char *ParallelG = (unsigned char*)malloc(widthRGB*heightRGB*channelRGB*sizeof(unsigned char));
	unsigned char *ParallelB = (unsigned char*)malloc(widthRGB*heightRGB*channelRGB*sizeof(unsigned char));
	unsigned char *ParallelGS = (unsigned char*)malloc(widthRGB*heightRGB*channelRGB*sizeof(unsigned char));
	unsigned char *ParallelBin = (unsigned char*)malloc(widthGS*heightGS*channelGS*sizeof(unsigned char));
	unsigned char *ParallelGrS = (unsigned char*)malloc(widthGS*heightGS*channelGS*sizeof(unsigned char));
	unsigned char *ParallelBini = (unsigned char*)malloc(widthBin*heightBin*channelGS*sizeof(unsigned char));

	cudaMalloc((void **) &deviceInputImageData, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageData1, width1 * height1 *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageDataRGB, widthRGB * heightRGB * channelRGB * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageDataGS, widthGS * heightGS * channelGS * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageDataBin, widthBin * heightBin * channelGS * sizeof(unsigned char));

	cudaMemcpy(deviceInputImageData, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageData1, image1, width1 * height1 * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageDataRGB, imageRGB, widthRGB * heightRGB * channelRGB * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageDataGS, imageGS, widthGS * heightGS * channelGS * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageDataBin, imageBin, widthBin * heightBin * channelGS * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &deviceOutputImageData, width * height *channels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageData1, (y2-y1+1)*(x2-x1+1) * channels * sizeof(unsigned char));	
	cudaMalloc((void **) &deviceOutputImageDataR, widthRGB * heightRGB *channelRGB * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageDataG, widthRGB * heightRGB *channelRGB * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageDataB, widthRGB * heightRGB *channelRGB * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageDataGS, widthRGB * heightRGB *channelRGB * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageDataBin, widthGS * heightGS *channelGS * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageDataGrS, widthGS * heightGS *channelGS * sizeof(unsigned char));
	cudaMalloc((void **) &deviceOutputImageDataBini, widthBin * heightBin *channelGS * sizeof(unsigned char));
	
	int* dev_sum_total, *dev_sum_square_total;
	cudaMalloc((void **) &dev_sum_total,  sizeof(int));
	cudaMalloc((void **) &dev_sum_square_total,  sizeof(int));
	cudaMemset(dev_sum_total,0,sizeof(int));
	cudaMemset(dev_sum_square_total,0,sizeof(int));
	
	dim3 dimGrid(ceil((float) width/TILE_WIDTH), ceil((float) height/TILE_WIDTH));
	//dim3 dimGrid2(ceil((float) width2/TILE_WIDTH), ceil((float) height2/TILE_WIDTH));
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
	
	//Contrast
	float *par_freq=(float*)malloc(256*sizeof(float));
	float *par_cdf=(float*)malloc(256*sizeof(float));
//	float time=paraContrast(image,Parallel,width,height,channels,par_freq,par_cdf);
//	cout<<" Contrast elapsed in time: "<<time<<" ms";
//	stbi_write_png("paraContrast.png",width,height,channels,Parallel,0);
	
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
	if(width==width1 && height==height1) 
    {
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
	}
	else
	{
		cout<< "ERROR! Images should be of same size for this Operation";
	}
	
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
	
	//Mean filter
	cout << "MeanFilter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_mean_filter<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraMeanFilter.png", width, height, channels, Parallel, 0); 
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Gaussian filter
	cout << "GaussianFilter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_gaussian_filter<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraGaussianFilter.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Median3 filter
	cout << "Median3Filter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_median_filter3<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraMedian3Filter.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Median5 filter
	cout << "Median5Filter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_median_filter5<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraMedian5Filter.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Max filter
	cout << "MaxFilter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_max_filter<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraMaxFilter.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Min filter
	cout << "MinFilter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_min_filter<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraMinFilter.png", width, height, channels, Parallel, 0);
	
	memset(Parallel,0,sizeof(width*height*channels*sizeof(unsigned char)));
	
	//Edge Detection 
	cout << "EdgeDetection elapsed in time: ";
	cudaEventRecord(startEvent);
	par_edge_detection<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, width, height, channels);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(Parallel, deviceOutputImageData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraEdgeDetection.png", width, height, channels, Parallel, 0);

	//Channel Split
	cout << "Channel Split elapsed in time: ";
	cudaEventRecord(startEvent);
	parchannelSplit<<<dimGrid,dimBlock>>>(deviceInputImageDataRGB, deviceOutputImageDataR, deviceOutputImageDataG, deviceOutputImageDataB, widthRGB, heightRGB);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime, startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelR, deviceOutputImageDataR, widthRGB * heightRGB * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_jpg("paraChannelS_R.png", widthRGB, heightRGB, 1, ParallelR, 0); 
	cudaMemcpy(ParallelG, deviceOutputImageDataG, widthRGB * heightRGB  * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_jpg("paraChannelS_G.png", widthRGB, heightRGB, 1, ParallelG, 0); 
	cudaMemcpy(ParallelB, deviceOutputImageDataB, widthRGB * heightRGB * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_jpg("paraChannelS_B.png", widthRGB, heightRGB, 1, ParallelB, 0); 

	//RGB to GrayScale 
	cout << "RGB to GrayScale elapsed in time: ";
	cudaEventRecord(startEvent);
	parRGBToGrey<<<dimGrid,dimBlock>>>(deviceInputImageDataRGB, deviceOutputImageDataGS, widthRGB, heightRGB);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelGS, deviceOutputImageDataGS, widthRGB * heightRGB * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("par_grey.png", widthRGB, heightRGB, 1, ParallelGS, 0);
	 
	//GrayScale to binary 
	cout << "GrayScale to binary elapsed in time: ";
	cudaEventRecord(startEvent);
	parGreyToBinary<<<dimGrid,dimBlock>>>(deviceInputImageDataGS, deviceOutputImageDataBin, widthGS, heightGS, 0.4);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelBin, deviceOutputImageDataBin, widthGS * heightGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("par_binary.png", widthGS, heightGS, 1, ParallelBin, 0);
	
	//Adaptive Noise Reduction Filter
	cout << "Adaptive Noise Reduction Filter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_adp_local_noise_reduction_filter<<<dimGrid,dimBlock>>>(deviceInputImageDataGS, deviceOutputImageDataGrS, widthGS, heightGS, channelGS, dev_sum_total, dev_sum_square_total);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelGrS, deviceOutputImageDataGrS, widthGS * heightGS * channelGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parADP_noise.png", widthGS, heightGS, channelGS, ParallelGrS, 0);
	
	memset(ParallelGrS,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));
	
	//Adaptive Median Filter
	cout << "Adaptive Median Filter elapsed in time: ";
	cudaEventRecord(startEvent);
	par_adp_median_filter<<<dimGrid,dimBlock>>>(deviceInputImageDataGS, deviceOutputImageDataGrS, widthGS, heightGS, channelGS);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelGrS, deviceOutputImageDataGrS, widthGS * heightGS * channelGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("parADP_median.png", widthGS, heightGS, channelGS, ParallelGrS, 0);

	memset(ParallelGrS,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));
	
	//Dialation Binary
	cout << "Dialation Binary elapsed in time: ";
	cudaEventRecord(startEvent);
	parDialation_binary<<<dimGrid,dimBlock>>>(deviceInputImageDataBin, deviceOutputImageDataBini, widthBin, heightBin);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelBini, deviceOutputImageDataBini, widthBin * heightBin * channelGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraDialationBinary.png", widthBin, heightBin, channelGS, ParallelBini, 0); 
	
	memset(ParallelBini,0,sizeof(widthBin*heightBin*channelGS*sizeof(unsigned char)));

	//Erode Binary
	cout << "Erode Binary elapsed in time: ";
	cudaEventRecord(startEvent);
	parErode_binary<<<dimGrid,dimBlock>>>(deviceInputImageDataBin, deviceOutputImageDataBini, widthBin, heightBin);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelBini, deviceOutputImageDataBini, widthBin * heightBin * channelGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraErodeBinary.png", widthGS, heightGS, channelGS, ParallelBini, 0);
	
	memset(ParallelBini,0,sizeof(widthBin*heightBin*channelGS*sizeof(unsigned char)));

	//Dialation Grey
	cout << "Dialation Grey elapsed in time: ";
	cudaEventRecord(startEvent);
	parDialation_Grey<<<dimGrid,dimBlock>>>(deviceInputImageDataGS, deviceOutputImageDataGrS, widthGS, heightGS);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelGrS, deviceOutputImageDataGrS, widthGS * heightGS * channelGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraDialationGrey.png", widthGS, heightGS, channelGS, ParallelGrS, 0); 

	memset(ParallelGrS,0,sizeof(widthGS*heightGS*channelGS*sizeof(unsigned char)));
	
	//Erode Grey
	cout << "Erode Grey elapsed in time: ";
	cudaEventRecord(startEvent);
	parErode_Grey<<<dimGrid,dimBlock>>>(deviceInputImageDataGS, deviceOutputImageDataGrS, widthGS, heightGS);
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&runTime,startEvent, stopEvent);
	cout<< runTime <<" ms"<<endl;
	cudaMemcpy(ParallelGrS, deviceOutputImageDataGrS, widthGS * heightGS * channelGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	stbi_write_png("paraErodeGrey.png", widthGS, heightGS, channelGS, ParallelGrS, 0);

    return 0;
}
