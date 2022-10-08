// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "../utils/Timer.hpp"
#include <iostream>
#include <cmath>
#include "../baseline/imgproc.hpp"
#include "../baseline/water.hpp"
#include "../utils/Histogram.hpp"
#include "water_cuda.hpp"
#include "imgproc_cuda.hpp"
std::shared_ptr<Image> runEnhanceStage(const Image *previous, const Histogram *hist, const WaterEffectOptions *options, Timer ts) {
  // Create a new image to store the result
  auto img_enhanced = std::make_shared<Image>(previous->width, previous->height);
  
  int numPixels = previous->height * previous->width;
  int numBlocks = (numPixels + 32 - 1) / 32;
  int blockSize = 32;

  // Move src image to device memory
  size_t img_size = sizeof(unsigned char) * numPixels * 4;
  unsigned char *src;
  cudaMallocManaged(&src, img_size);
  cudaMemcpy((void *)src, (void *)(previous->raw.data()), img_size, cudaMemcpyHostToDevice);

  ts.start();

  // Determine the threshold from the histogram, by taking 10% of the maximum value in the histogram.
  auto max_hist = (int) (hist->max(0) * 0.1);

  unsigned char begin = 0;
  unsigned char end = 255;

  // Enhance each (non-alpha) channel of the source image
  for (int i = 0; i < 3; i++)
  {
    // Obtain the first intensity that is above the threshold.
    for (begin = 0; begin < hist->range; begin++) {
      if (hist->count(begin, i) > max_hist) {
        break;
      }
    }

    // Obtain the last intensity that is above the threshold.
    for (end = 255; end > begin; end--) {
      if (hist->count(end, i) > max_hist) {
        break;
      }
    }
    enhanceContrastLinearlyCuda<<<numBlocks, blockSize>>>(src, src, begin, end, i, numPixels);
    // Wait for completion
    cudaDeviceSynchronize();
    cudaGetLastError();
  }
  ts.stop();

  std::cout << "Stage: Contrast enhance:        " << ts.seconds() << " s." << std::endl;

  // Transfer enhanced image from device to host memory
  cudaMemcpy(img_enhanced->raw.data(), (void *)src, img_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(src);

  // Save the resulting image
  if (options->save_intermediate)
    img_enhanced->toPNG("output/" + options->img_name + "_enhancedCUDA.png");

  // Create and save the enhanced histogram (if enabled).
  if (options->enhance_hist) {
    auto enhanced_hist = getHistogram(img_enhanced.get());
    auto enhanced_hist_img = enhanced_hist.toImage();
    enhanced_hist_img->toPNG("output/" + options->img_name + "_enhanced_histogramCUDA.png");
  }

  return img_enhanced;
}
std::shared_ptr<Histogram> runHistogramStage(const Image *previous, const WaterEffectOptions *options, Timer ts) {
    // Histogram to hold result
    // Histogram *hist_res = new Histogram();
    auto hist_res = std::make_shared<Histogram>();

    // Determine # of threads to allocate
    int numPixels = previous->height * previous->width;
    int numBlocks = (numPixels + 32 - 1) / 32;
    int blockSize = 32;

    // Histogram stage
    
    if (options->histogram) {
   	 
	// Move src 
    size_t img_size = sizeof(unsigned char) * numPixels * 4;
    unsigned char *src;
    cudaMallocManaged(&src, img_size);
    //move to device
    cudaMemcpy((void *)src, (void *)(previous->raw.data()), img_size, cudaMemcpyHostToDevice);

    // Set up device memory for histogram
    int *hist;
    size_t hist_size = sizeof(int) * 4 * 256;
    cudaMallocManaged(&hist, hist_size);
    cudaMemset(hist, 0, hist_size);

    ts.start();
    getHistogramCuda<<<numBlocks, blockSize>>>(src, numPixels, hist);
    // Wait for completion
    cudaDeviceSynchronize();
    ts.stop();
    std::cout << "Stage: Histogram:        " << ts.seconds() << " s." << std::endl;

    // Copy the result data back to host
    cudaMemcpy(hist_res->values.data(), hist, hist_size, cudaMemcpyDeviceToHost);

    if (options->save_intermediate) {
      // Copy raw data into histogram object
      auto hist_img = hist_res->toImage();
      hist_img->toPNG("output/" + options->img_name + "_histogramCUDA.png");
    }
    cudaFree(src);
    cudaFree(hist); 
  }
  return hist_res;
}

std::shared_ptr<Image> runWaterEffectCUDA(const Image *src, const WaterEffectOptions *options) {
  /* REPLACE THIS CODE WITH YOUR OWN WATER EFFECT PIPELINE */
Timer ts;
  // Smart pointers to intermediate images:
  std::shared_ptr<Histogram> hist;
  std::shared_ptr<Image> img_result;
  // Histogram stage
  if (options->histogram)
  {
    hist = runHistogramStage(src, options, ts);
  }
  if (options->enhance)
  {
    if (hist == nullptr) {
      throw std::runtime_error("Cannot run enhance stage without histogram.");
    }
    img_result = runEnhanceStage(src, hist.get(), options, ts);
  }
  return nullptr;
  /* REPLACE THIS CODE WITH YOUR OWN WATER EFFECT PIPELINE */
}
