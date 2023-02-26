/**
 * @file
 * @brief convolution blurring in Nvidia CUDA
 * @author Arjun31415
 */

#undef __noinline__
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

cv::cuda::GpuMat ginput, goutput;

/**
 * @brief do a safe call to CUDA functions and handle the error along with user
 * specified message
 *
 * @param err CUDA error code
 * @param msg user specified message
 * @param file_name the name of the file from which the error occurred
 * @param line_number the line at which error occurred
 */
static inline void _safe_cuda_call(cudaError err, const char *msg,
								   const char *file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",
				msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}
/**
 * @brief a macro for sage calling CUDA functions
 * @param call the CUDA function call
 * @param msg user specified message
 */

#define SAFE_CALL(call, msg) _safe_cuda_call((call), (msg), __FILE__, __LINE__)

/**
 * @brief generate the gaussian kernel with given kernel size and standard
 * deviation
 *
 * @param kernel the array in which the weights are stored
 * @param n the size of the kernel, t.e. n x n kernel is created
 * @param sigma  the standard deviation
 */
__host__ void generate_gaussian_kernel_2d(float *kernel, const int n,
										  const float sigma = 1)
{
	int mean = n / 2;
	float sumOfWeights = 0;
	float p, q = 2.0 * sigma * sigma;

	// Compute weights
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			p = sqrt((i - mean) * (i - mean) + (j - mean) * (j - mean));
			kernel[i * n + j] = std::exp((-(p * p) / q)) / (M_PI * q);
			sumOfWeights += kernel[i * n + j];
		}
	}

	// Normalizing weights
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			kernel[i * n + j] /= sumOfWeights;
		}
	}
}

__host__ void generate_gaussian_kernel_1d(float *kernel, const int n,
										  const float sigma = 1)
{
	// Calculate the values of the kernel
	float sum = 0.0f;
	for (int i = 0; i < n; i++)
	{
		float x = i - (n - 1) / 2.0f;
		kernel[i] = std::exp(-x * x / (2 * sigma * sigma));
		sum += kernel[i];
	}

	// Normalize the kernel so that its sum equals 1
	for (int i = 0; i < n; i++)
	{
		kernel[i] /= sum;
	}
}

/**
 * @brief      Sets the value of a uchar type.
 *
 * @param  val   The value
 * @param      out   The output
 */
__device__ __forceinline__ void set_value(const int &val, uchar &out)
{
	out = val;
}

/**
 * @brief set the value for a floating point type.
 *
 * @param val  the value
 * @param out the output
 */
__device__ __forceinline__ void set_value(const float &val, float &out)
{
	out = val;
};

/**
 * @brief set the value for a float3 tupe. All the 3 fields will have the value
 * `val`
 *
 * @param val the value
 * @param out the output
 */
__device__ __forceinline__ void set_value(const float &val, float3 &out)
{
	out.x = val, out.y = val, out.z = val;
}

/**
 * @brief set the value for a unsigned char3 type with a flot3 type
 *
 * @param val the value to set
 * @param out the ouput
 */
__device__ __forceinline__ void set_value(const float3 &val, uchar3 &out)
{
	out.x = val.x;
	out.y = val.y;
	out.z = val.z;
}
/**
 * @brief      Sets the value of a uchar3 type.
 *
 * @param[in]  val   The value
 * @param      out   The output
 */
__device__ __forceinline__ void set_value(const int &val, uchar3 &out)
{
	out.x = val;
	out.y = val;
	out.z = val;
}

/**
 * @brief      Subtraction for uchar3 types.
 *
 * @param[in]  in1   Input 1
 * @param[in]  in2   Input 2
 *
 * @return     Output
 */
__device__ __forceinline__ uchar3 subtract_value(uchar3 in1, uchar3 in2)
{
	uchar3 out;
	out.x = in1.x - in2.x;
	out.y = in1.y - in2.y;
	out.z = in1.z - in2.z;
	return out;
}

/**
 * @brief add two values and return it
 *
 * @param in1 input 1
 * @param in2 intput 2
 * @return returns the added value
 */
__device__ __forceinline__ float3 add_value(float3 in1, float3 in2)
{
	return {in1.x + in2.x, in1.y + in2.y, in1.z + in2.z};
}

/**
 * @brief add two floating point values
 *
 * @param in1 value 1
 * @param in2 value 2
 * @return the sum
 */
__device__ __forceinline__ float add_value(float in1, float in2)
{
	return in1 + in2;
}

/**
 * @brief      Subtraction for uchar types.
 *
 * @param[in]  in1   Input 1
 * @param[in]  in2   Input 2
 *
 * @return     Output
 */
__device__ __forceinline__ uchar subtract_value(uchar in1, uchar in2)
{
	return in1 - in2;
}
/**
 * @brief multiplication for float and uchar3 types. Multiply each filed in
 * uchar3 with the float value and return a flolat3
 *
 * @param x Input 1
 * @param y Input 2
 * @return value after multiplication
 */
__device__ __forceinline__ float3 multiply_value(const float &x,
												 const uchar3 &y)
{
	return {x * (float)y.x, x * (float)y.y, x * (float)y.z};
}

/**
 * @brief multiplication for float and float3 types. Multiply each filed in
 * uchar3 with the float value and return a float3
 *
 * @param x Input 1
 * @param y Input 2
 * @return value after multiplication
 */
__device__ __forceinline__ float3 multiply_value(const float &x,
												 const float3 &y)
{
	return {x * (float)y.x, x * (float)y.y, x * (float)y.z};
}

/**
 * @brief multiplication for float and uchar4 types
 *
 * @param x Input 1
 * @param y Input 2
 * @return x*y
 */
__device__ __forceinline__ float multiply_value(const float &x, const uchar &y)
{
	return x * (float)y;
}

/**
 * @brief applys the gaussian blur convolution to the input image
 *
 * @tparam T_in The type of input image, i.e uchar for black and white, uchar3
 for RGB, float3 etc
 * @tparam T_out The type of output image
 * @tparam F_cal The type for calculating intermediate sums and products
 * @param kernel the kernel to apply the convolution
 * @param n the dimension of the kernel (n x n)
 * @param input the input image
 * @param output the output image
 */
template <typename T_in, typename T_out, typename F_cal>
__global__ void gaussian_blur(const float *kernel, int n,
							  const cv::cuda::PtrStepSz<T_in> input,
							  cv::cuda::PtrStepSz<T_out> output)
{
	// calculate the x & y position of the current image pixel
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= input.cols || y >= input.rows) return;

	const int mid = n / 2;
	F_cal sum;
	set_value(0, sum);
	// synchronize all the threads till this potin
	__syncthreads();

	// loop over the n x n neighborhood of the current pixel
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int y_idx = y + i - mid;
			int x_idx = x + j - mid;
			if (y_idx > input.rows || x_idx > input.cols) continue;
			const float kernel_val = kernel[(n - i - 1) * n + (n - j - 1)];
			sum =
				add_value(sum, multiply_value(kernel_val, input(y_idx, x_idx)));
		}
	}
	T_out result;
	set_value(sum, result);
	output(y, x) = result;
}
template <typename T_in, typename T_out, typename F_cal>
__global__ void gaussian_blur_x(float *kernel, int kernel_size,
								const cv::cuda::PtrStepSz<T_in> input,
								cv::cuda::PtrStepSz<T_out> output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int radius = kernel_size / 2;
	const int width = input.cols;
	const int height = input.rows;

	if (x >= input.cols || y >= input.rows) return;

	F_cal pixel;
	set_value(0, pixel);

	for (int i = -radius; i <= radius; i++)
	{
		int idx = y * width + (x + i);
		/* printf("%d\n", idx); */
		if (idx >= 0 && idx < width * height)
		{
			const float weight = kernel[i + radius];
			pixel = add_value(pixel, multiply_value(weight, input[idx]));
		}
	}
	set_value(pixel, output(y, x));
}

template <typename T_in, typename T_out, typename F_cal>

__global__ void gaussian_blur_y(float *kernel, int kernel_size,
								const cv::cuda::PtrStepSz<T_in> input,
								cv::cuda::PtrStepSz<T_out> output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int radius = kernel_size / 2;
	const int width = input.cols;
	const int height = input.rows;

	if (x >= input.cols || y >= input.rows) return;

	F_cal pixel;
	set_value(0, pixel);
	float weight_sum = 0;
	for (int i = -radius; i <= radius; i++)
	{
		int idx = (y + i) * width + x;
		if (idx >= 0 && idx < width * height)
		{
			float weight = kernel[i + radius];
			pixel = add_value(pixel, multiply_value(weight, input[idx]));
		}
	}
	set_value(pixel, output(y, x)); //	output(y,x) = pixel;
}

/**
 * @brief free all the GPU resources
 *
 * @tparam Ts
 * @param inputs varidaic list of resources
 */
template <class... Ts>
void gaussian_blur_exit(Ts &&...inputs)
{
	ginput.release();
	goutput.release();
	([&] { SAFE_CALL(cudaFree(inputs), "Unable to free"); }(), ...);
}

/**
 * @brief calls the gaussian_blur function appropriately based on the type of
 * image
 *
 * @param d_kernel the kernel, stored on GPU device memory
 * @param n the size of the kernel
 * @param input the input image stored on the GPU
 * @param output the output image stored on the GPU
 */
void call_gaussian_blur_2d(float *d_kernel, const int &n,
						   const cv::cuda::GpuMat &input,
						   cv::cuda::GpuMat &output)
{
	CV_Assert(input.channels() == 1 || input.channels() == 3);
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(input.cols, block.x),
					cv::cuda::device::divUp(input.rows, block.y));
	if (input.channels() == 1)
	{
		gaussian_blur<uchar, uchar, float>
			<<<grid, block>>>(d_kernel, n, input, output);
		return;
	}
	else if (input.channels() == 3)
	{
		gaussian_blur<uchar3, uchar3, float3>
			<<<grid, block>>>(d_kernel, n, input, output);
	}
	cudaSafeCall(cudaGetLastError());
}
void call_gaussian_blur_1d(float *d_kernel, const int &n,
						   const cv::cuda::GpuMat &input,
						   cv::cuda::GpuMat &output)
{
	CV_Assert(input.channels() == 1 || input.channels() == 3);
	const int block_size = 16;
	dim3 dimBlock(block_size, block_size);
	dim3 dimGrid(cv::cuda::device::divUp(input.cols, dimBlock.x),
				 cv::cuda::device::divUp(input.rows, dimBlock.y));
	cv::cuda::GpuMat temp = input.clone();
	// Apply the horizontal Gaussian blur
	if (input.channels() == 1)
	{

		gaussian_blur_x<uchar, uchar, float>
			<<<dimGrid, dimBlock>>>(d_kernel, n, input, temp);
		gaussian_blur_y<uchar, uchar, float>
			<<<dimGrid, dimBlock>>>(d_kernel, n, temp, output);
	}
	else if (input.channels() == 3)
	{
		gaussian_blur_x<uchar3, uchar3, float3>
			<<<dimGrid, dimBlock>>>(d_kernel, n, input, temp);
		gaussian_blur_y<uchar3, uchar3, float3>
			<<<dimGrid, dimBlock>>>(d_kernel, n, temp, output);
	}
	cudaSafeCall(cudaGetLastError());
}
/**
 * @brief the gaussian blur function which runs on the HOST CPU. It calls the
 * `call_gaussian_blur` function after initialization of the appropriate values
 * and kernel.
 *
 * @param input the input image stored on the CPU memory
 * @param output the output image stored on the CPU memory
 * @param n the size of the Gaussian kernel, defaults to 3
 * @param sigma the standard deviation of the Gaussian kernel, defaults to 1.
 */
__host__ void gaussian_blur(const cv::Mat &input, cv::Mat &output,
							const int n = 3, const float sigma = 1.0,
							bool two_d = true)
{
	ginput.upload(input);
	std::vector<float> gauss_kernel_host;
	float *d_gauss_kernel;
	if (two_d)
	{
		std::cout << "2D" << std::endl;
		gauss_kernel_host = std::vector<float>(n * n);
		generate_gaussian_kernel_2d(gauss_kernel_host.data(), n, sigma);
		cudaMalloc((void **)&d_gauss_kernel, n * n * sizeof(float));
		SAFE_CALL(cudaMemcpy(d_gauss_kernel, gauss_kernel_host.data(),
							 sizeof(float) * n * n, cudaMemcpyHostToDevice),
				  "Unable to copy kernel");
		call_gaussian_blur_2d(d_gauss_kernel, n, ginput, goutput);
	}
	else
	{
		std::cout << "1D" << std::endl;

		gauss_kernel_host = std::vector<float>(n);
		generate_gaussian_kernel_1d(gauss_kernel_host.data(), n, sigma);
		cudaMalloc((void **)&d_gauss_kernel, n * sizeof(float));
		SAFE_CALL(cudaMemcpy(d_gauss_kernel, gauss_kernel_host.data(),
							 sizeof(float) * n, cudaMemcpyHostToDevice),
				  "Unable to copy kernel");
		call_gaussian_blur_1d(d_gauss_kernel, n, ginput, goutput);
	}
	// goutput.upload(input);
	goutput.download(output);
	gaussian_blur_exit(d_gauss_kernel);
}

/**
 * @brief initialization for gaussian blurring operation
 *
 * @param input input image stored on the CPU
 * @param output output image stored on the CPU
 */
void gaussian_blur_init(const cv::Mat &input, cv::Mat &output)
{
	ginput.create(input.rows, input.cols, input.type());
	goutput.create(output.rows, output.cols, output.type());
}

int main(int argc, char **argv)
{

	if (argc < 3)
	{
		printf("usage: Blur_Test <kernel_size> <Image_Path> [<Output_Path>]\n");
		return -1;
	}
	std::string mTitle = "Display Image";
	cv::Mat input;
	int n = atoi(argv[1]);
	input = cv::imread(argv[2], 1);
	if (!input.data)
	{
		printf("No image data \n");
		return -1;
	}
	assert(input.channels() == 3);
	auto output = input.clone();

	// Call the wrapper function
	gaussian_blur_init(input, output);
	gaussian_blur(input, output, n, 1.7, 0);

	// Show the input and output
	cv::imshow("Output", output);

	// Wait for key press
	cv::waitKey();
	namedWindow(mTitle, cv::WINDOW_AUTOSIZE);
	/* namedWindow("gauss", cv::WINDOW_AUTOSIZE); */
	imshow(mTitle, input);
	/* imshow("gaussian", output); */
	if (argc >= 4) imwrite(argv[3], output);
	do
	{

		auto k = cv::waitKey(500);
		if (k == 27)
		{
			cv::destroyAllWindows();
			return 0;
		}
		if (cv::getWindowProperty(mTitle, cv::WND_PROP_VISIBLE) == 0) return 0;

	} while (true);

	return 0;
}
