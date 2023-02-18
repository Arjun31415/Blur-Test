#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
void generate_gaussian_kernel(std::vector<std::vector<float>> &kernel,
							  const int n, const float sigma = 1)
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
			kernel[i][j] = std::exp((-(p * p) / q)) / (M_PI * q);
			sumOfWeights += kernel[i][j];
		}
	}

	// Normalizing weights
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			kernel[i][j] /= sumOfWeights;
		}
	}
}
void apply_convolution(const std ::vector<std::vector<float>> &kernel,
					   const Mat &original_img, Mat &new_img, const int &r,
					   const int &c)
{
	const size_t n = kernel.size();
	assert(n % 2 == 1);
	assert(n == kernel[0].size());
	const size_t mid = n / 2;
	new_img.at<Vec3b>(r, c) = {0, 0, 0};
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (r - mid + i >= 0 && r - mid + i < original_img.rows &&
				c - mid + j >= 0 && c - mid + j < original_img.cols)
				new_img.at<Vec3b>(r, c) +=
					kernel[n - i - 1][n - j - 1] *
					original_img.at<Vec3b>(r - mid + i, c - mid + j);
		}
	}
}
void apply_convolution_multi_threaded(
	const std::vector<std::vector<float>> &kernel, const Mat &original_img,
	Mat &new_img, const int &r, const int &c)
{
	const size_t n = kernel.size();
	assert(n % 2 == 1);
	assert(n == kernel[0].size());
	const size_t mid = n / 2;
	new_img.at<Vec3b>(r, c) = {0, 0, 0};
#pragma omp parallel for shared(r, c, original_img, new_img, kernel)
	for (int i = 0; i < n; i++)
	{
#pragma omp parallel for shared(r, c, original_img, new_img, kernel)
		for (int j = 0; j < n; j++)
		{
			if (r - mid + i >= 0 && r - mid + i < original_img.rows &&
				c - mid + j >= 0 && c - mid + j < original_img.cols)
				new_img.at<Vec3b>(r, c) +=
					kernel[n - i - 1][n - j - 1] *
					original_img.at<Vec3b>(r - mid + i, c - mid + j);
		}
	}
}
void apply_kernel(const std::vector<std::vector<float>> &kernel,
				  const Mat &original_img, Mat &new_img)
{
	for (int i = 0; i < original_img.rows; i++)
	{
		for (int j = 0; j < original_img.cols; j++)
		{
			apply_convolution(kernel, original_img, new_img, i, j);
		}
	}
}
void apply_kernel_multithreaded(const std::vector<std::vector<float>> &kernel,
								const Mat &original_img, Mat &new_img)
{
#pragma omp parallel for shared(original_img, new_img, kernel)
	for (int i = 0; i < original_img.rows; i++)
	{
#pragma omp parallel for shared(original_img, new_img, kernel)
		for (int j = 0; j < original_img.cols; j++)
		{
			apply_convolution(kernel, original_img, new_img, i, j);
		}
	}
}
int main(int argc, char **argv)
{
	/* for (auto &x : gauss_kernel) {
		for (auto &y : x)
		{
			std::cout << y << " ";
		}
		std::cout << "\n";
	} */

	if (argc < 3)
	{
		printf("usage: Blur_Test <kernel_size> <Image_Path> [<Output_Path>]\n");
		return -1;
	}
	int n = atoi(argv[1]);
	std::vector<std::vector<float>> gauss_kernel(n, std::vector<float>(n));
	generate_gaussian_kernel(gauss_kernel, n, 1.6);

	std::string mTitle = "Display Image";
	Mat image;
	image = imread(argv[2], 1);
	if (!image.data)
	{
		printf("No image data \n");
		return -1;
	}
	namedWindow(mTitle, WINDOW_AUTOSIZE);
	auto new_img = image.clone();
	// namedWindow("gauss", WINDOW_AUTOSIZE);
	apply_kernel_multithreaded(gauss_kernel, image, new_img);
	imshow(mTitle, image);
	imshow("gaussian", new_img);
	if (argc >= 4) imwrite(argv[3], new_img);
	do
	{

		auto k = waitKey(500);
		if (k == 27)
		{
			cv::destroyAllWindows();
			return 0;
		}
		if (cv::getWindowProperty(mTitle, WND_PROP_VISIBLE) == 0)
		{
			return 0;
			break;
		}

	} while (true);
	return 0;
}
