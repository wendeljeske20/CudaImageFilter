
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"
#include <math.h>  

#include <iostream>
#include <string>
using namespace std;
using namespace cimg_library;


void ProcessImage(unsigned int *defaultColor, unsigned int *newColor, int imageWidth, int imageHeight, float *filter, int filterWidth, float filterFactor);

__host__ __device__ void PackColorBits(unsigned int &color, int r, int g, int b)
{
	color = ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | (b & 0x0ff);
}

__host__ __device__ void UnpackColorBits(int rgb, int &r, int &g, int &b)
{
	r = (rgb >> 16) & 0x0ff;
	g = (rgb >> 8) & 0x0ff;
	b = (rgb) & 0x0ff;
}

void HostConvolutionFilter(unsigned int *defaultColor, unsigned int *newColor, int imageWidth, int imageHeight, float *filter, int filterWidth, float filterFactor)
{
	int r, g, b;

	const int midX = floor(float(filterWidth));
	const int midY = floor(float(filterWidth));

	//float factor = 1.0;
	float bias = 0;



	//apply the filter
	int imageSize = imageWidth * imageHeight;
	for (int i = 0; i < imageSize; i++)
	{


		float red = 0.0, green = 0.0, blue = 0.0;


		for (int k = 0; k < filterWidth * filterWidth; k++)
		{

			//printf("for");
			int filterX;
			if (k > 0)
				filterX = k % filterWidth;
			else
				filterX = 0;

			int filterY;
			if (k > 0)
				filterY = floor(float(k) / filterWidth);
			else
				filterY = 0;


			int pixelOffset = filterX - midX + imageWidth * (filterY - midY);
			int index = i + pixelOffset;

			//int index = min(max(i + pixelOffset, 0), 1360 * 7 - 1);
			//int index = 1360 * ();
			if (index >= 0 && index < imageSize)
			{
				UnpackColorBits(defaultColor[index], r, g, b);
				//cout << r << " ";
				red += r * filter[k];
				green += g * filter[k];
				blue += b * filter[k];
			}
		}

		//truncate values smaller than zero and larger than 255
		r = min(max(int(filterFactor * red + bias), 0), 255);
		g = min(max(int(filterFactor * green + bias), 0), 255);
		b = min(max(int(filterFactor * blue + bias), 0), 255);

		//defaultColor[i] = 200;
		PackColorBits(newColor[i], r, g, b);
	}
}

__global__ void DeviceConvolutionFilter(unsigned int *defaultColor, unsigned int *newColor, int *imageWidth, int *imageHeight, float *filter, int *filterWidth, float *filterFactor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int r, g, b;

	const int midX = floor(float(*filterWidth));
	const int midY = floor(float(*filterWidth));

	float bias = 0;

	float red = 0.0, green = 0.0, blue = 0.0;

	int filterSize = *filterWidth * *filterWidth;

	for (int k = 0; k < filterSize; k++)
	{

		//printf("for");
		int filterX;
		if (k > 0)
			filterX = k % *filterWidth;
		else
			filterX = 0;

		int filterY;
		if (k > 0)
			filterY = floor(float(k) / *filterWidth);
		else
			filterY = 0;


		int pixelOffset = filterX - midX + *imageWidth * (filterY - midY);
		int index = i + pixelOffset;
		//int index = 1360 * ();

		UnpackColorBits(defaultColor[index], r, g, b);
		red += r * filter[k];
		green += g * filter[k];
		blue += b * filter[k];

	}

	//truncate values smaller than zero and larger than 255
	r = min(max(int(*filterFactor * red + bias), 0), 255);
	g = min(max(int(*filterFactor * green + bias), 0), 255);
	b = min(max(int(*filterFactor * blue + bias), 0), 255);


	PackColorBits(newColor[i], r, g, b);
}







void SetImageColor(CImg<unsigned char> &image, unsigned int *color, int width, int height)
{
	int k = 0;

	for (int y = 0; y < height; y++)
	{

		for (int x = 0; x < width; x++)
		{
			//cor[y * width + x] = 225125050;
			//newImage[y * width + x] = 2000;

			int r, g, b;
			UnpackColorBits(color[k], r, g, b);

			image(x, y, 0) = r;
			image(x, y, 1) = g;
			image(x, y, 2) = b;
			//if(k < 5000)
			k++;
		}
	}
}

void SetColorArray(unsigned int *color, CImg<unsigned char> image, int width, int height)
{
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			PackColorBits(color[y * width + x], image(x, y, 0), image(x, y, 1), image(x, y, 2));
		}
	}
}

int main()
{

	int imageWidth = 1360;
	int imageHeight = 768;

	CImg<unsigned char> image1("printdoinvicto.bmp");
	CImg<unsigned char> image2("astroneer.bmp");
	CImg<unsigned char> newImage(imageWidth, imageHeight, 1, 3, 255);// = image;

	

	float motionBlurFilter[] =
	{
	  1, 0, 0, 0, 0, 0, 0, 0, 0,
	  0, 1, 0, 0, 0, 0, 0, 0, 0,
	  0, 0, 1, 0, 0, 0, 0, 0, 0,
	  0, 0, 0, 1, 0, 0, 0, 0, 0,
	  0, 0, 0, 0, 1, 0, 0, 0, 0,
	  0, 0, 0, 0, 0, 1, 0, 0, 0,
	  0, 0, 0, 0, 0, 0, 1, 0, 0,
	  0, 0, 0, 0, 0, 0, 0, 1, 0,
	  0, 0, 0, 0, 0, 0, 0, 0, 1,
	};

	float edgeEnhancementFilter[] =
	{
		 0,-1, 0,
		-1, 1, 0,
		 0, 0, 0
	};

	float sharpenFilter[] =
	{
		 0,-1, 0,
		-1, 5,-1,
		 0,-1, 0
	};

	float embossFilter[] =
	{
		-2,-1, 0,
		-1, 1, 1,
		 0, 1, 2
	};

	const int filterWidth = 3;
	float *currentFilter = sharpenFilter;

	float filterFactor = 1.0;


	//as cores são armazenas em um array de inteiros unidimensional usando empacotamento de bits
	unsigned int *color = new unsigned int[imageWidth * imageHeight];
	unsigned int *newColor = new unsigned int[imageWidth * imageHeight];

	for (int i = 0; i < imageWidth * imageHeight; i++)
	{
		newColor[i] = 100;
	}

	//SetColorArray(color, image1, width, height);
	//ProcessImage(color, width, height, filter, filterWidth, filterFactor);
	//SetImageColor(newImage, color, width, height);

	CImgDisplay window(imageWidth, imageHeight);

	CImg<unsigned char> imgtext;
	unsigned char clr[] = { 255,0,0 };




	while (!window.is_closed())
	{


		if (window.is_keyESC())
			window.close();


		if (window.is_keyARROWRIGHT())
		{
			filterFactor *= 1.1f;
			//printf("%f ",filterFactor);
		}
		else if (window.is_keyARROWLEFT())
		{
			filterFactor *= 0.9f;
			//printf("%f ", filterFactor);
		}


		if (window.is_keyC())
		{
			SetColorArray(color, image1, imageWidth, imageHeight);
			HostConvolutionFilter(color, newColor, imageWidth, imageHeight, currentFilter, filterWidth, filterFactor);
			SetImageColor(newImage, newColor, imageWidth, imageHeight);
		}

		if (window.is_keyG())
		{
			SetColorArray(color, image2, imageWidth, imageHeight);
			ProcessImage(color, newColor, imageWidth, imageHeight, currentFilter, filterWidth, filterFactor);
			SetImageColor(newImage, newColor, imageWidth, imageHeight);

		}






		window.display(newImage);

		/*string txt = to_string(filterFactor);
		imgtext.draw_text(2000, 2000, txt.c_str(), clr, 0, 1, 23);
		window.display(imgtext);*/

		//window.
		//imgtext.clear();
		window.wait();
	}


	//newImage.save("printdoinvictocomfiltro.bmp");


	return 0;
}




void ProcessImage(unsigned int *defaultColor, unsigned int *newColor, int imageWidth, int imageHeight, float *filter, int filterWidth, float filterFactor)
{

	unsigned int *deviceColor = 0;
	unsigned int *deviceNewColor = 0;
	int *deviceImageWidth = 0;
	int *deviceImageHeight = 0;
	float *deviceFilter = 0;
	int *deviceFilterWidth = 0;
	float *deviceFilterFactor = 0;


	unsigned int imageSize = imageWidth * imageHeight;


	// Allocate GPU buffers 
	cudaMalloc((void**)&deviceColor, imageSize * sizeof(int));
	cudaMalloc((void**)&deviceNewColor, imageSize * sizeof(int));
	cudaMalloc((void**)&deviceImageWidth, sizeof(int));
	cudaMalloc((void**)&deviceImageHeight, sizeof(int));

	cudaMalloc((void**)&deviceFilter, filterWidth * filterWidth * sizeof(float));
	cudaMalloc((void**)&deviceFilterWidth, sizeof(int));
	cudaMalloc((void**)&deviceFilterFactor, sizeof(int));


	// Copy from host memory to GPU buffers.
	cudaMemcpy(deviceColor, defaultColor, imageSize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceImageWidth, &imageWidth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceImageHeight, &imageHeight, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(deviceFilter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceFilterWidth, &filterWidth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceFilterFactor, &filterFactor, sizeof(int), cudaMemcpyHostToDevice);


	// Launch a kernel on the GPU with one thread for each element.
	DeviceConvolutionFilter << <imageSize / 512, 512 >> > (deviceColor, deviceNewColor, deviceImageWidth, deviceImageWidth, deviceFilter, deviceFilterWidth, deviceFilterFactor);


	cudaDeviceSynchronize();

	// Copy from GPU buffer to host memory.
	cudaMemcpy(newColor, deviceNewColor, imageSize * sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(deviceColor);
	cudaFree(deviceNewColor);
	cudaFree(deviceImageWidth);
	cudaFree(deviceImageHeight);
	cudaFree(deviceFilter);
	cudaFree(deviceFilterWidth);
	cudaFree(deviceFilterFactor);

}


