/**
 * @file
 * @brief gaussian blurring using CPU
 * @author Arjun31415
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace cv;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

/**
 * @brief Generate a 2D gaussian kernel
 *
 * @param kernel the kernel to be populated
 * @param n the size of the kernel, the `kernel` must be of size $n * n$
 * @param sigma the standard deviation of the gaussian kernel
 */
void generate_gaussian_kernel(std::vector<std::vector<float>> &kernel,
                              const int n, const float sigma = 1) {
  int mean = n / 2;
  float sumOfWeights = 0;
  float p, q = 2.0 * sigma * sigma;

  // Compute weights
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      p = sqrt((i - mean) * (i - mean) + (j - mean) * (j - mean));
      kernel[i][j] = std::exp((-(p * p) / q)) / (M_PI * q);
      sumOfWeights += kernel[i][j];
    }
  }

  // Normalizing weights
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      kernel[i][j] /= sumOfWeights;
    }
  }
}
/**
 * @brief apply a convolution kernel to a pixel
 *
 * @param kernel the convolution kernel
 * @param original_img the original image
 * @param new_img the output image
 * @param r the row number of the current pixel
 * @param c the column number of the current pixel

 */
void apply_convolution(const std ::vector<std::vector<float>> &kernel,
                       const Mat &original_img, Mat &new_img, const int &r,
                       const int &c) {
  const size_t n = kernel.size();
  assert(n % 2 == 1);
  assert(n == kernel[0].size());
  const size_t mid = n / 2;
  new_img.at<Vec3b>(r, c) = {0, 0, 0};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (r - mid + i >= 0 && r - mid + i < original_img.rows &&
          c - mid + j >= 0 && c - mid + j < original_img.cols)
        new_img.at<Vec3b>(r, c) +=
            kernel[n - i - 1][n - j - 1] *
            original_img.at<Vec3b>(r - mid + i, c - mid + j);
    }
  }
}
/**
 * @brief apply a convolution kernel to a pixel using multiple threads (OMP)
 *
 * @param kernel the convolution kernel
 * @param original_img the original image
 * @param new_img the output image
 * @param r the row number of the current pixel
 * @param c the column number of the current pixel
 */
void apply_convolution_multi_threaded(
    const std::vector<std::vector<float>> &kernel, const Mat &original_img,
    Mat &new_img, const int &r, const int &c) {
  const size_t n = kernel.size();
  assert(n % 2 == 1);
  assert(n == kernel[0].size());
  const size_t mid = n / 2;
  new_img.at<Vec3b>(r, c) = {0, 0, 0};
#pragma omp parallel for shared(r, c, original_img, new_img, kernel)
  for (int i = 0; i < n; i++) {
#pragma omp parallel for shared(r, c, original_img, new_img, kernel)
    for (int j = 0; j < n; j++) {
      if (r - mid + i >= 0 && r - mid + i < original_img.rows &&
          c - mid + j >= 0 && c - mid + j < original_img.cols)
        new_img.at<Vec3b>(r, c) +=
            kernel[n - i - 1][n - j - 1] *
            original_img.at<Vec3b>(r - mid + i, c - mid + j);
    }
  }
}
/**
 * @brief apply a convolution kernel to the entire image
 *
 * @param kernel the convolution kernel
 * @param original_img the original image
 * @param new_img the output image
 */
void apply_kernel(const std::vector<std::vector<float>> &kernel,
                  const Mat &original_img, Mat &new_img) {
  for (int i = 0; i < original_img.rows; i++) {
    for (int j = 0; j < original_img.cols; j++) {
      apply_convolution(kernel, original_img, new_img, i, j);
    }
  }
}
/**
 * @brief apply a convolution kernel to the entire image using multiple threads
 * (OMP)
 *
 * @param kernel  the convolution kernel
 * @param original_img the original_img
 * @param new_img the output image
 */
void apply_kernel_multithreaded(const std::vector<std::vector<float>> &kernel,
                                const Mat &original_img, Mat &new_img) {
#pragma omp barrier
#pragma omp parallel for shared(original_img, new_img, kernel)
  for (int i = 0; i < original_img.rows; i++) {
#pragma omp parallel for shared(original_img, new_img, kernel)
    for (int j = 0; j < original_img.cols; j++) {
      apply_convolution(kernel, original_img, new_img, i, j);
    }
  }
#pragma omp barrier
}
void stress_test(const int &n, const bool &multi = true) {
  std::cout << "Stress testing" << std::endl;
  const std::string path = "../images/peppers_color.tif";
  auto image = imread(path, 1);
  auto new_img = image.clone();
  std::vector<std::vector<float>> gauss_kernel(n, std::vector<float>(n));
  generate_gaussian_kernel(gauss_kernel, n, 1.6);
  std::vector<double> run_times;
  const int num_runs = 20;
  std::string fname;
  if (!multi) {
    fname = "profile_single_threaded.csv";

    for (int i = 0; i < num_runs; i++) {
      printProgress((float)i / num_runs);
      auto start = std::chrono::high_resolution_clock::now();
      apply_kernel(gauss_kernel, image, new_img);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration_ms = end - start;
      run_times.push_back(duration_ms.count());
    }
  } else {
    fname = "profile_multi_threaded.csv";
    for (int i = 0; i < num_runs; i++) {
      printProgress((float)i / num_runs);
      auto start = std::chrono::high_resolution_clock::now();
      apply_kernel_multithreaded(gauss_kernel, image, new_img);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration_ms = end - start;
      run_times.push_back(duration_ms.count());
    }
  }
  sort(run_times.begin(), run_times.end());
  double avg = std::accumulate(run_times.begin(), run_times.end(), 0.0) /
               run_times.size();
  double median =
      ((run_times.size() % 2 == 0) ? (run_times[run_times.size() / 2 - 1] +
                                      run_times[run_times.size() / 2]) /
                                         2
                                   : run_times[run_times.size() / 2]);

  std::cout << "\nMax(ms)\t\tAvg(ms)\t\tMedian(ms)\t\tMin(ms)\n";
  std::cout << run_times.back() << "\t\t" << avg << "\t\t" << median

            << "\t\t" << run_times.front() << "\n";
  std::fstream file(fname, std::ios::in | std::ios_base::app);
  if (file.tellg() == 0) {
    // write the headers if the file is empty
    file << "KERNEL_SIZE,MAX_RUN_TIME,MIN_RUN_TIME,AVG_RUN_TIME,MEDIAN_RUN_"
            "TIME"
         << std::endl;
  }
  file << n << "," << run_times.back() << "," << run_times.front() << "," << avg
       << "," << median << std::endl;

  file.close();
}
int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: Blur_Test <kernel_size> <Image_Path> [<Output_Path>]\n");
    return -1;
  }
  int n = atoi(argv[1]);
  if (strncmp(argv[2], "stressm", 7) == 0) {
    stress_test(n, true);
    return 0;
  } else if (strncmp(argv[2], "stress", 6) == 0) {
    stress_test(n, false);
    return 0;
  }

  std::vector<std::vector<float>> gauss_kernel(n, std::vector<float>(n));
  generate_gaussian_kernel(gauss_kernel, n, 1.6);

  std::string mTitle = "Display Image";
  Mat image;
  image = imread(argv[2], 1);
  if (!image.data) {
    printf("No image data \n");
    return -1;
  }
  namedWindow(mTitle, WINDOW_AUTOSIZE);
  auto new_img = image.clone();
  apply_kernel_multithreaded(gauss_kernel, image, new_img);
  imshow(mTitle, image);
  imshow("gaussian", new_img);
  if (argc >= 4)
    imwrite(argv[3], new_img);
  do {

    auto k = waitKey(500);
    if (k == 27) {
      cv::destroyAllWindows();
      return 0;
    }
    if (cv::getWindowProperty(mTitle, WND_PROP_VISIBLE) == 0) {
      return 0;
      break;
    }

  } while (true);
  return 0;
}
