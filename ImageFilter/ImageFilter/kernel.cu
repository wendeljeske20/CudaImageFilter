
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"
#include <math.h>  

#include <iostream>
using namespace std;
using namespace cimg_library;


void ProcessImage(unsigned int *color, unsigned int size);

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



__global__ void Filter(unsigned int *srcColor, unsigned int *dstColor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	int r, g, b;
	UnpackColorBits(srcColor[i], r, g, b);

	//r = 50;
	//g = 50;
	//b = 255;


	PackColorBits(dstColor[i], r, g, b);
}

__global__ void Filter2(unsigned int *srcColor, unsigned int *dstColor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;


	int r, g, b;

	const int filterWidth = 5;
	const int filterHeight = 5;

	float filter[filterWidth][filterHeight] =
	{
		0,  0, -1,  0,  0,
		0,  0, -1,  0,  0,
		0,  0,  2,  0,  0,
		0,  0,  0,  0,  0,
		0,  0,  0,  0,  0,
	};

	const int midX = floor(float(filterWidth));
	const int midY = floor(float(filterHeight));

	float factor = 1.0;
	float bias = 0;

	int w = 1360;
	int h = 768;

	//apply the filter

	double red = 0.0, green = 0.0, blue = 0.0;

	//multiply every value of the filter with corresponding image pixel
	for (int filterY = 0; filterY < filterWidth; filterY++)
	{
		for (int filterX = 0; filterX < filterHeight; filterX++)
		{


			int index = i + filterX - midX + w * (filterY - midY);

			UnpackColorBits(srcColor[index], r, g, b);
			red += r * filter[filterY][filterX];
			green += g * filter[filterY][filterX];
			blue += b * filter[filterY][filterX];
		}
	}

	//truncate values smaller than zero and larger than 255
	r = min(max(int(factor * red + bias), 0), 255);
	g = min(max(int(factor * green + bias), 0), 255);
	b = min(max(int(factor * blue + bias), 0), 255);


	PackColorBits(dstColor[i], r, g, b);
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

	int width = 1360;
	int height = 768;

	CImg<unsigned char> image1("printdoinvicto.bmp");
	CImg<unsigned char> image2("astroneer.bmp");
	CImg<unsigned char> newImage(width, height, 1, 3, 255);// = image;


	int filter[3][3] =
	{
	   0, 0, 0,
	   0, 1, 0,
	   0, 0, 0
	};

	unsigned int *color = new unsigned int[width * height];




	SetColorArray(color, image1, width, height);
	ProcessImage(color, width * height);
	SetImageColor(newImage, color, width, height);

	CImgDisplay window(width, height);

	while (!window.is_closed())
	{
		if (window.is_keyESC())
			window.close();

		if (window.is_keySPACE())
		{
			SetColorArray(color, image2, width, height);
			ProcessImage(color, width * height);
			SetImageColor(newImage, color, width, height);
			cout << "apertou" << endl;
		}



		window.display(newImage);

		window.wait();
	}


	//newImage.save("printdoinvictocomfiltro.bmp");

	//std::system("pause");
	return 0;
}




void ProcessImage(unsigned int *color, unsigned int size)
{
	unsigned int *dev_color = 0;
	unsigned int *dev_new_color = 0;



	// Allocate GPU buffers 
	cudaMalloc((void**)&dev_color, size * sizeof(int));
	cudaMalloc((void**)&dev_new_color, size * sizeof(int));

	// Copy from host memory to GPU buffers.
	cudaMemcpy(dev_color, color, size * sizeof(int), cudaMemcpyHostToDevice);


	// Launch a kernel on the GPU with one thread for each element.
	Filter2 << <size / 512, 512 >> > (dev_color, dev_new_color);


	cudaDeviceSynchronize();

	// Copy from GPU buffer to host memory.
	cudaMemcpy(color, dev_new_color, size * sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_color);
	cudaFree(dev_new_color);

}


