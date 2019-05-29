
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"

#include <iostream>
using namespace std;
using namespace cimg_library;


void ProcessImage(unsigned int *color, unsigned int size);

__host__ __device__ void PackageColor(unsigned int &color, int r, int g, int b)
{
	color = ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | (b & 0x0ff);
}

__host__ __device__ void DecomposeColor(int rgb, int &r, int &g, int &b)
{
	r = (rgb >> 16) & 0x0ff;
	g = (rgb >> 8) & 0x0ff;
	b = (rgb) & 0x0ff;
}



__global__ void Filter(unsigned int *srcColor, unsigned int *dstColor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	int r, g, b;
	DecomposeColor(srcColor[i], r, g, b);

	//r = 50;
	//g = 50;
	//b = 255;


	PackageColor(dstColor[i], r, g, b);
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
			DecomposeColor(color[k], r, g, b);

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
			 PackageColor(color[y * width + x],	image(x, y, 0), image(x, y, 1), image(x, y, 2));
		}
	}
}

int main()
{
	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//addWithCuda(c, a, b, arraySize);


	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaDeviceReset();





	int width = 1360;
	int height = 768;

	CImg<unsigned char> image1("printdoinvicto.bmp");
	CImg<unsigned char> image2("astroneer.bmp");
	CImg<unsigned char> newImage(width, height, 1, 3, 255);// = image;


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
	Filter << <1360, 768 >> > (dev_color, dev_new_color);


	cudaDeviceSynchronize();

	// Copy from GPU buffer to host memory.
	cudaMemcpy(color, dev_new_color, size * sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(dev_color);
	cudaFree(dev_new_color);

}


