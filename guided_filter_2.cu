/*#include "guided_filter.h"

template<typename G_Type>
__global__ void meanSmooth(G_Type *input_data, G_Type *output_data, G_Type *input_data_2, G_Type* output_data_2 ,int window_size, int image_width, int image_height) {

	const int pixel_x = blockDim.x*blockIdx.x + threadIdx.x;
	const int pixel_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (pixel_x >= image_width-window_size || pixel_y >= image_height-window_size) return;

	G_Type out_pixel = 0 , out_pixel_2 =0;

	const int element_index = pixel_x + pixel_y*image_width;

	uint8_t window_sum = window_size*window_size;

	int w_count = window_size / 2;
	for (int32_t m_h = -w_count; m_h < w_count; m_h++) {
		for (int32_t m_w = -w_count; m_w < w_count; m_w++)
		{
			out_pixel += input_data[pixel_x + m_w + (m_h + pixel_y)*image_width] / window_sum;
			out_pixel_2 += input_data_2[pixel_x + m_w + (m_h + pixel_y)*image_width] / window_sum;
		}
	}
	output_data[element_index] = out_pixel;
	output_data_2[element_index] = out_pixel_2;
}

template<typename G_Type>
__global__ void MatElementMul(G_Type *f_data, G_Type *s_data, G_Type *out_data, int32_t mat_width, int32_t mat_height) {

	const int mat_x = blockDim.x*blockIdx.x + threadIdx.x;
	const int mat_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (mat_x >= mat_width || mat_y >= mat_height) return;
	const int elem_index = mat_y*mat_height + mat_x;

	out_data[elem_index] = f_data[elem_index] * s_data[elem_index];
	
}

template<typename G_Type>
__global__ void ProcessAB(G_Type *mean_I, G_Type *mean_II,
	G_Type *mean_p, G_Type mean_Ip, int image_width, int image_height, 
	float *a_data, float *b_data, float eps) {

	const int mat_x = blockDim.x*blockIdx.x + threadIdx.x;
	const int mat_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (mat_x >= image_width || mat_y >= image_height) return;

	const int element_index = mat_y*image_width + mat_x;

	
	float a_result = float(mean_Ip[element_index]) / (mean_II[element_index] - mean_I[element_index] * mean_I[element_index] + eps);
	a_data[element_index] = a_result;
	b_data[element_index] = float(mean_p[element_index] - a_result*mean_I[element_index]);
}

template<typename G_Type>
__global__ void ProcessImage(G_Type *guided_data, G_Type *smooth_data, float *a_data,
	float* b_data,int image_width, int image_height) 
{
	int mat_x = blockDim.x*blockIdx.x + threadIdx.x;
	int mat_y = blockDim.y*blockIdx.y + threadIdx.y;

	if (mat_x >= image_width || mat_y >= image_height) 
		return;
	int element_index = mat_y*image_width + mat_x;

	smooth_data[element_index] = a_data[element_index] * guided_data[element_index] + b_data[element_index];
}

template<typename G_Type>
void GuidedImage(G_Type *guided_data, G_Type *depth_data, G_Type*& smooth_data,
	int image_width,int image_height,int window_size)
{
	G_Type *guided_filter_data, *depth_filter_data;
	int data_size = image_height*image_width * sizeof(G_Type);
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

	G_Type *guided_mul_data, depth_mul_data;
	cudaMallocManaged((void**)&guided_mul_data, data_size);
	cudaMallocManaged((void**)&depth_mul_data, data_size);

	MatElementMul << <grid_dim, thread_dim >> > (guided_filter_data, guided_filter_data, guided_mul_data, image_width, image_height);
	MatElementMul << <grid_dim, thread_dim >> > (guided_filter_data, depth_data, depth_mul_data, image_width, image_height);

	float *a_data, *b_data;
	cudaMallocManaged((void**)&a_data, image_width*image_height * sizeof(float));
	cudaMallocManaged((void**)&b_data, image_width*image_height * sizeof(float));
	ProcessAB << < grid_dim, thread_dim >> > (guided_filter_data, depth_filter_data, guided_mul_data, depth_mul_data, image_width, image_height);

	ProcessImage << < grid_dim, thread_dim >> > (guided_data, smooth_data, a_data, b_data,image_width,image_height);

}*/


