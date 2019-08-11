#include"guided_filter.h"

__global__ void warm_up() {
	int c = 0;
}
int main(int argc, char* argv[]) {

	
	assert(argc > 1);
	string guided_path = argv[1];
	string disp_path = argv[2];

	std::cout<<guided_path<<std::endl;
	const int window_size = atoi(argv[3]);
	//float eps = atoi(argv[4]);

	cv::Mat guided_image = cv::imread(guided_path, 0);
	cv::Mat disp_image = cv::imread(disp_path, 0);
        cout<<disp_image.size()<<endl;
	assert(guided_image.size() == disp_image.size()||guided_image.empty()||disp_image.empty());
	float *guided_data, *disp_data, *smooth_data;

	int image_width = guided_image.cols;
	int image_height = guided_image.rows;
	cudaMallocManaged((void**)&guided_data, image_width*image_height * sizeof(float));
	cudaMallocManaged((void**)&disp_data, image_width*image_height*(sizeof(float)));
	cudaMallocManaged((void**)&smooth_data, image_width*image_height*(sizeof(float)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	for (int img_w = 0; img_w < image_width; img_w++) {
		for (int img_h = 0; img_h < image_height; img_h++) {
			guided_data[img_h*image_width + img_w] = guided_image.at<uint8_t>(img_h, img_w);
			disp_data[img_h*image_width + img_w] = disp_image.at<uint8_t>(img_h, img_w);
			//uint8_t diap_num = guided_image.at<uint8_t>(img_h, img_w);
			//printf("%u\n", diap_num);
		}
	}
	
	warm_up << <100, 1000 >> >();

	cudaEventRecord(start, 0);
	GuidedImage(guided_data, disp_data, smooth_data, image_width, image_height, window_size,0.21);
	//GuidedImage(guided_data, disp_data, smooth_data, image_width, image_height, window_size, 0.2);
        //GuidedImage(guided_data, disp_data, smooth_data, image_width, image_height, window_size, 0.2);
	cudaEventRecord(stop, 0);

	cudaDeviceSynchronize();

	cudaEventSynchronize(start);

	cudaEventSynchronize(stop);

	float time_elapsed = 0;
	cudaEventElapsedTime(&time_elapsed, start, stop);

	printf("time£º%f(ms)\n", time_elapsed);

	cv::Mat new_result_image(disp_image.size(),CV_8UC1);
	for (int img_h = 0; img_h < image_height; img_h++) {
		for (int img_w = 0; img_w < image_width; img_w++) {

			new_result_image.at<uint8_t>(img_h, img_w) = int(smooth_data[img_w + img_h*image_width]);
			//printf("%u\n", new_result_image.at<uint8_t>(img_h, img_w));
		}
	}
        cout<<disp_image.size()<<endl;
	cv::imwrite("filter.png", new_result_image);
        cv::waitKey(0);
	for (;;);

}
