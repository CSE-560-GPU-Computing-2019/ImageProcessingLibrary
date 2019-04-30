#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<ctime>
using namespace cv;
using namespace std;


void HistogramEqualization(Mat image)
{
	//change the color image to grayscale image
    cvtColor(image, image, COLOR_BGR2GRAY); 

    //equalize the histogram
    Mat hist_equalized_image;
	clock_t begin =clock();
    equalizeHist(image, hist_equalized_image); 
 	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" HIstogram equalization OPENCV "<<elapsedTime<<" ms\n";
 imwrite( "HEopencv.png", hist_equalized_image );   
}

void Flip(Mat image, int orientation)
{
	Mat FlippedImage;
	clock_t begin =clock();
	flip(image, FlippedImage, orientation); 
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	
	
	if(orientation==0)
	{
		cout<<" FLIP HORIZONTAL OPENCV "<<elapsedTime<<" ms\n";
		imwrite( "FlipHorizontalopencv.png", FlippedImage );  
	}
	else if(orientation==1)
	{
		cout<<" FLIP VERTICAL OPENCV "<<elapsedTime<<" ms\n";
		imwrite( "FlipVerticalopencv.png", FlippedImage );
	}
}
void Rotate(Mat image, int direction)
{
	Mat rotate;
	clock_t begin =clock();
	transpose(image, rotate);
	 flip(rotate, rotate, direction);
	 clock_t end =clock();
	 double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	if(direction==0)
	{
		cout<<" ANTI CLOCKWISE OPENCV "<<elapsedTime<<" ms\n";
		imwrite( "antiClockwiseopencv.png", rotate );
	}
	else if(direction==1)
	{
		cout<<" CLOCKWISE OPENCV "<<elapsedTime<<" ms\n";
		imwrite( "Clockwiseopencv.png", rotate );
	}
   
}

void Crop(Mat image, int topleft_x, int topleft_y, int cropwidth, int cropheight,int width,int height)
{
	if(topleft_x<width && topleft_y<height && (topleft_x+cropwidth)<width && (topleft_y+cropheight)<height)
	{
		clock_t begin =clock();
		Rect Cropped= Rect(topleft_x,topleft_y,cropwidth,cropheight);
	    Mat CroppedImage = image(Cropped);
		 clock_t end =clock();
		  double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
		  cout<<" CROP OPENCV "<<elapsedTime<<" ms\n";
	   imwrite( "Croppedopencv.png", CroppedImage );
	}
	
}	
void Inverse(Mat image)
{
	clock_t begin =clock();
	cv::Mat invSrc =  cv::Scalar::all(255) - image;
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" INVERSE OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "inverseopencv.png", invSrc );
}

void Not(Mat image)
{
 	clock_t begin =clock();
	cv::Mat invSrc =  cv::Scalar::all(255) - image;
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" NOT  OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "Notopencv.png", invSrc );
}
void Brightness(Mat image, int intensity)
{
	Mat bright;
	clock_t begin =clock();
	image.convertTo(bright, -1, 1, intensity); //increase the brightness by 50
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" BRIGHTNESS OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "Brightnessopencv.png", bright );
	
	 
}
void BitwiseAnd(cv::Mat image, cv::Mat image1)
{
//	cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";
	Mat res;
//	Mat res = Mat::zeros( Size(400,200), CV_8UC3 );
//	Mat drawing1 = Mat::zeros( Size(image.cols,image.rows), CV_8UC1 );
 //       Mat drawing2 = Mat::zeros( Size(image1.cols,image1.rows), CV_8UC1 );
 //       drawing1(Range(0,drawing1.rows),Range(0,drawing1.cols))=255;
  //      drawing2(Range(0,drawing1.rows),Range(0,drawing2.cols))=255;
	clock_t begin =clock();
	bitwise_and(image,image1,res); 
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	//cout<<" BITWISE AND OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "BitwiseAndopencv.png", res );
	//cout<<" *************************************** \n";
}
void BitwiseOR(Mat image, Mat image2)
{
	Mat res;
	clock_t begin =clock();
	bitwise_or(image,image2,res); 
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" BITWISE OR OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "BitwiseOropencv.png", res );
}
void BitwiseXOR(Mat image, Mat image2)
{
	Mat res;
	clock_t begin =clock();
	bitwise_xor(image,image2,res); 
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" BITWISE  XOR OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "BitwiseXoropencv.png", res );
}

void GaussianBlur(Mat image)
{
	Mat image_blurred_with_5x5_kernel;
	clock_t begin =clock();
    GaussianBlur(image, image_blurred_with_5x5_kernel, Size(5, 5), 0);
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" GAUSSIAN OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "gaussianopencv.png", image_blurred_with_5x5_kernel );
	
}
void Erosion(Mat image)
{
	Mat image_eroded_with_5x5_kernel;
	clock_t begin =clock();
    erode(image, image_eroded_with_5x5_kernel, getStructuringElement(MORPH_RECT, Size(3, 3)));
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" erosion OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "Erosionopencv.png", image_eroded_with_5x5_kernel );
	cout<<" min filter OPENCV "<<elapsedTime <<" ms\n";
}

void Dilate(Mat image)
{
	Mat image_dilated_with_5x5_kernel;
	clock_t begin =clock();
    erode(image, image_dilated_with_5x5_kernel, getStructuringElement(MORPH_RECT, Size(3, 3)));
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" Dilate OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "Dilateopencv.png", image_dilated_with_5x5_kernel );
	cout <<" Max Filter OPENCV"<< elapsedTime<< " ms\n";
}

void MeanFilter(Mat image, int kszie)
{
	Mat res;
	clock_t begin =clock();
	blur( image, res, Size( 3, 3 ), Point(-1,-1) );
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" Mean Filter OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "MeanFilteropencv.png", res );
}
void MedianFilter (Mat image, int ksize)
{
	Mat res;
	clock_t begin =clock();
	medianBlur( image, res, 3);
	clock_t end =clock();
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" Median Filter OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "MedianFilteropencv.png", res );
}
void RGBtoGrayScale(Mat image)
{
	Mat res;
	clock_t begin =clock();
	cvtColor(image, res, COLOR_BGR2GRAY);
	clock_t end =clock();	
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" RGB ->GRAY OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "Grayopencv.png", res );
}
void GrayScaletoBinary(Mat image,int thres)
{
	Mat res;
	clock_t begin =clock();
	threshold(image, res, thres, 255,THRESH_BINARY);
	clock_t end =clock();	
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" GRAY ->BINARY OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "binaryopencv.png", res );
}
void edgeDetection(Mat image)
{
	 Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	 Mat grad;
  
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
	clock_t begin =clock();
	GaussianBlur(image, image, Size(5, 5), 0);
	cvtColor( image, image, CV_BGR2GRAY );
	 
	/// Gradient X
	Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	
	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	clock_t end =clock();	
	double elapsedTime =double(end-begin)/(CLOCKS_PER_SEC/1000);
	cout<<" EdgeDetection OPENCV "<<elapsedTime<<" ms\n";
	imwrite( "edgeDetection.png", grad );
}

void ChannelSplit(Mat image)
{
   Mat bgr[3];
   clock_t begin= clock();
   split(image,bgr);
   clock_t end =clock();
   float elapsedTime=double(end-begin)/(CLOCKS_PER_SEC/1000);
 cout<<" ChannelSplit OPENCV "<< elapsedTime <<" ms";
   imwrite("blue.png",bgr[0]);
   imwrite("red.png",bgr[2]);
   imwrite("green.png",bgr[1]); 
}
int main(int argc, char** argv)
{
    // Read the image file
    Mat image = imread("lena1024.png");

	Mat image2 = imread("lenaFlip1024.png");
   Mat image3=imread("Grayopencv1024.png");
    //imwrite("input_copy.png",image);

//	Mat drawing1 = Mat::zeros( Size(image.cols,image.rows), CV_8UC3 );
 //   Mat drawing2 = Mat::zeros( Size(image2.cols,image2.rows), CV_8UC3 );
	
	//drawing1(Range(0,drawing1.rows),Range(0,drawing1.cols/2))=255; 
    //drawing2(Range(100,150),Range(150,350))=255;
	
    // Check for failure
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

	HistogramEqualization(image3);// gives grayscale output
	Flip(image,0);
	Flip(image,1);
        Rotate(image,0);
	Rotate(image,1);
	Crop(image,0,0,128,512,image.cols,image.rows);
	Inverse(image);
	Not(image);
	Brightness(image,50);
    	BitwiseAnd(image,image2);
	BitwiseOR(image,image2);
	BitwiseXOR(image,image2);
	GaussianBlur(image);
	Erosion(image3);
	Dilate(image3);
	MeanFilter(image,3);
	MedianFilter(image,3);
	edgeDetection(image);
	RGBtoGrayScale(image);
//	Mat image4= imread("Grayopencv512.png");
	GrayScaletoBinary(image3,127);
	ChannelSplit(image);
    return 0;
}







