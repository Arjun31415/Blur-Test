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
__host__ void generate_gaussian_kernel(float *kernel, const int n,
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
__device__ __forceinline__ void set_value(const float &val, float3 &out)
{
	out.x = val, out.y = val, out.z = val;
}
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
__device__ __forceinline__ float3 add_value(float3 in1, float3 in2)
{
	return {in1.x + in2.x, in1.y + in2.y, in1.z + in2.z};
}
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
__device__ __forceinline__ float3 multiply_value(const float &x,
												 const uchar3 &y)
{
	return {x * (float)y.x, x * (float)y.y, x * (float)y.z};
}
__device__ __forceinline__ float multiply_value(const float &x, const uchar &y)
{
	return x * (float)y;
}
__device__ __forceinline__ void print_value(const uchar3 &x)
{
	printf("(%d, %d, %d)", x.x, x.y, x.z);
}
__device__ __forceinline__ void print_value(const uchar &x) { printf("%d", x); }
__device__ __forceinline__ void print_value(const float3 &x)
{
	printf("(%f, %f, %f)", x.x, x.y, x.z);
}
__device__ __forceinline__ void print_value(const uchar &x, const uchar &y)
{
	printf("(%d, %d)", x, y);
}
__device__ __forceinline__ void print_value(const uchar3 &x, const uchar3 &y)
{
	printf("((%d,%d,%d), (%d,%d,%d))\n", x.x, x.y, x.z, y.x, y.y, y.z);
}
template <typename T_in, typename T_out, typename F_cal>
__global__ void gaussian_blur(const float *kernel, int n,
							  const cv::cuda::PtrStepSz<T_in> input,
							  cv::cuda::PtrStepSz<T_out> output)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= input.cols || y >= input.rows) return;
	const int mid = n / 2;
	F_cal sum;
	set_value(0, sum);
	__syncthreads();

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
template <class... Ts>
void gaussian_blur_exit(Ts &&...inputs)
{
	ginput.release();
	goutput.release();
	([&] { SAFE_CALL(cudaFree(inputs), "Unable to free"); }(), ...);
}

__device__ void print_kernel(float *k, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%f ", k[i * n + j]);
		}
		printf("\n");
	}
}
void call_gaussian_blur(float *d_kernel, const int &n,
						const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output)
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
__host__ void gaussian_blur(const cv::Mat &input, cv::Mat &output,
							const int n = 3, const float sigma = 1.0)
{

	std::vector<float> gauss_kernel_host(n * n);
	generate_gaussian_kernel(gauss_kernel_host.data(), n, sigma);
	/* for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << gauss_kernel_host[i * n + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cin.get(); */
	float *d_gauss_kernel;
	cudaMalloc((void **)&d_gauss_kernel, n * n * sizeof(float));
	SAFE_CALL(cudaMemcpy(d_gauss_kernel, gauss_kernel_host.data(),
						 sizeof(float) * n * n, cudaMemcpyHostToDevice),
			  "Unable to copy kernel");
	ginput.upload(input);
	goutput.upload(input);
	call_gaussian_blur(d_gauss_kernel, n, ginput, goutput);
	goutput.download(output);
	gaussian_blur_exit(d_gauss_kernel);
}
void gaussian_blur_init(const cv::Mat &input, cv::Mat &output)
{
	ginput.create(input.rows, input.cols, input.type());
	goutput.create(output.rows, output.cols, output.type());
}

int main(int argc, char **argv)
{

	if (argc < 2)
	{
		printf("usage: Blur_Test <Image_Path> [<Output_Path>]\n");
		return -1;
	}
	std::string mTitle = "Display Image";
	cv::Mat input;
	input = cv::imread(argv[1], 1);
	if (!input.data)
	{
		printf("No image data \n");
		return -1;
	}
	assert(input.channels() == 3);
	auto output = input.clone();

	// Call the wrapper function
	gaussian_blur_init(input, output);
	gaussian_blur(input, output, 5, 1.7);

	// Show the input and output
	cv::imshow("Output", output);

	// Wait for key press
	cv::waitKey();
	namedWindow(mTitle, cv::WINDOW_AUTOSIZE);
	/* namedWindow("gauss", cv::WINDOW_AUTOSIZE); */
	imshow(mTitle, input);
	/* imshow("gaussian", output); */
	if (argc >= 3) imwrite(argv[2], output);
	do
	{

		auto k = cv::waitKey(500);
		if (k == 27)
		{
			cv::destroyAllWindows();
			return 0;
		}
		std::cout << cv::getWindowProperty(mTitle, cv::WND_PROP_VISIBLE)
				  << "\n";
		if (cv::getWindowProperty(mTitle, cv::WND_PROP_VISIBLE) == 0) return 0;

	} while (true);

	return 0;
}
