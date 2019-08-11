#pragma once
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;


__global__ void meanSmooth(float *input_data, float *output_data, float *input_data_2, float* output_data_2, int window_size, int image_width, int image_height) {

	const int pixel_x = blockDim.x*blockIdx.x + threadIdx.x;
	const int pixel_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (pixel_x < window_size||pixel_x>=image_width-window_size || pixel_y >= image_height - window_size||pixel_y<window_size) return;

	float out_pixel = 0.0f, out_pixel_2 = 0.0f;

	const int element_index = pixel_x + pixel_y*image_width;

	int window_sum = (2*window_size+1)*(2*window_size+1);

	int w_count = window_size ;
	for (int m_h = -w_count; m_h <= w_count; m_h++) {
		for (int m_w = -w_count; m_w <= w_count; m_w++)
		{
			//printf("%d\n",out_pixel);
			out_pixel += input_data[pixel_x + m_w + (m_h + pixel_y)*image_width];
			out_pixel_2 += input_data_2[pixel_x + m_w + (m_h + pixel_y)*image_width];
		}
	}
	out_pixel /= (window_sum);
	out_pixel_2 /= (window_sum);
	//printf("%d\n", out_pixel);
	output_data[element_index] = out_pixel;
	output_data_2[element_index] = out_pixel_2;

	//printf("%u::%u\n", output_data[element_index], output_data_2[element_index]);
}

__global__ void MatElementMul(float *f_data, float *s_data, float *out_data, int mat_width, int mat_height) {

	const int mat_x = blockDim.x*blockIdx.x + threadIdx.x;
	const int mat_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (mat_x >= mat_width || mat_y >= mat_height) return;
	const int elem_index = mat_y*mat_width + mat_x;

	out_data[elem_index] = f_data[elem_index] * s_data[elem_index];
	//printf("%f\n", out_data[elem_index]);

}

__global__ void ProcessAB(float *mean_I, float *mean_p,
	float *mean_II, float *mean_Ip, int image_width, int image_height,
	float *a_data, float *b_data, float eps) {

	const int mat_x = blockDim.x*blockIdx.x + threadIdx.x;
	const int mat_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (mat_x >= image_width || mat_y >= image_height) return;

	const int element_index = mat_y*image_width + mat_x;

	//printf("%u::%u::%u\n", mean_Ip[element_index], mean_II[element_index], mean_I[element_index]);
	float a_result = (mean_Ip[element_index]- mean_I[element_index] * mean_p[element_index]) / (mean_II[element_index] - mean_I[element_index] * mean_I[element_index] + eps);
	a_data[element_index] = a_result;
	b_data[element_index] = float(mean_p[element_index] - a_result*mean_I[element_index]);
	//printf("%f\n", a_data[element_index]);
}


__global__ void ProcessImage(float *guided_data, float *smooth_data, float *a_data,
	float* b_data, int image_width, int image_height)
{
	int mat_x = blockDim.x*blockIdx.x + threadIdx.x;
	int mat_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (mat_x >= image_width || mat_y >= image_height)
		return;
	int element_index = mat_y*image_width + mat_x;

	int smooth_color = a_data[element_index] * int(guided_data[element_index]) + b_data[element_index];
	if (smooth_color > 255) smooth_color = 255;
	if (smooth_color < 0) smooth_color = 0;
	smooth_data[element_index] = smooth_color;
	//printf("thread::%u::%u::%f::%f\n", smooth_data[element_index], guided_data[element_index],a_data[element_index],b_data[element_index]);
}

void GuidedImage(float *guided_data, float *depth_data, float* smooth_data,
	int image_width, int image_height, int window_size,float eps)
{
	float *guided_filter_data, *depth_filter_data;
	int data_size = image_height*image_width * sizeof(float);
	cudaMallocManaged((void**)&guided_filter_data, data_size);
	cudaMallocManaged((void**)&depth_filter_data, data_size);

	const int thread_num = 16;
	dim3 thread_dim;
	thread_dim.x = thread_num;
	thread_dim.y = thread_num;

	dim3 grid_dim;
	grid_dim.x = image_width / thread_num + 1;
	grid_dim.y = image_height / thread_num + 1;
	meanSmooth << < grid_dim, thread_dim >> > (guided_data, guided_filter_data, depth_data, depth_filter_data, window_size, image_width, image_height);
	//meanSmooth << < grid_dim, thread_dim >> > (guided_data, guided_filter_data, depth_data, smooth_data, window_size, image_width, image_height);

	float *guided_mul_data, *depth_mul_data;

	int mul_size = image_width*image_height * sizeof(float);
	cudaMallocManaged((void**)&guided_mul_data, mul_size);
	cudaMallocManaged((void**)&depth_mul_data, mul_size);

	MatElementMul << <grid_dim, thread_dim >> > (guided_data, guided_data, guided_mul_data, image_width, image_height);
	MatElementMul << <grid_dim, thread_dim >> > (guided_data, depth_data, depth_mul_data, image_width, image_height);

	float *guided_mul_filter_data, *depth_mul_filter_data;

	cudaMallocManaged((void**)&guided_mul_filter_data, mul_size);
	cudaMallocManaged((void**)&depth_mul_filter_data, mul_size);
	meanSmooth << <grid_dim, thread_dim >> > (guided_mul_data, guided_mul_filter_data, depth_mul_data, depth_mul_filter_data, window_size, image_width, image_height);
	
	float *a_data, *b_data,*a_filter_data,*b_filter_data;
	cudaMallocManaged((void**)&a_data, image_width*image_height * sizeof(float));
	cudaMallocManaged((void**)&b_data, image_width*image_height * sizeof(float));

	cudaMallocManaged((void**)&a_filter_data, image_width*image_height * sizeof(float));
	cudaMallocManaged((void**)&b_filter_data, image_width*image_height * sizeof(float));

	ProcessAB << < grid_dim, thread_dim >> > (guided_filter_data, depth_filter_data,guided_mul_filter_data,depth_mul_filter_data, image_width, image_height,a_data,b_data,eps);

	meanSmooth << <grid_dim, thread_dim >> > (a_data, a_filter_data, b_data, b_filter_data, window_size, image_width, image_height);

	ProcessImage << < grid_dim, thread_dim >> > (guided_data, smooth_data, a_filter_data, b_filter_data, image_width, image_height);

}





/*template<typename G_Type>
__global__ void ProcessImage(G_Type *guided_data, G_Type *smooth_data, float *a_data,
	float* b_data, int image_width, int image_height);*/
