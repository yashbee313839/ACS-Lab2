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

#include "imgproc_cuda.hpp"

__global__ void getHistogramCuda(const unsigned char *src, int numPixels, int *hist)
{
    assert((src != nullptr));
    //loop to go throough every pixel
    int threadCount = threadIdx.x + (blockIdx.x * blockDim.x);
    //int temp = 4*threadCount;
    for (;threadCount < numPixels; threadCount += blockDim.x * gridDim.x)
    {
	int temp = 4 * threadCount;
        unsigned char rchi = src[temp]; // intensity of red channel
	atomicAdd(&hist[(0) + rchi], 1);
        unsigned char gchi = src[temp+1]; // intensity of green channel
	atomicAdd(&hist[256 + gchi], 1);
        unsigned char bchi = src[temp+2]; // intensity of blue channel
	atomicAdd(&hist[(2*256) + bchi], 1);
        unsigned char achi = src[temp+3]; // intensity of alpha channel
	atomicAdd(&hist[(3* 256) + achi], 1);
    }
}
__global__ void enhanceContrastLinearlyCuda(unsigned char *src, unsigned char *dest, 
                                        unsigned char first, unsigned char last, 
                                        int channel, int numPixels)
{
    assert((src != nullptr) && (dest != nullptr));

    int threadCount = threadIdx.x + (blockIdx.x * blockDim.x);
    //int temp = 4*threadCount;
    for (;threadCount < numPixels; threadCount += blockDim.x * gridDim.x)
    {
        int temp = (4 * threadCount) + channel; // coordinate in images for this thread
        if (src[temp] < first) {
            dest[temp] = 0;
        } else if (src[temp] > last) {
            dest[temp] = 255;
        } else {
            // Anything else is scaled
            dest[temp] = (unsigned char) ((255.0f/(last-first)) * (src[temp] - first));
        }
    }    
}

/*__global__ void applyRippleCuda(unsigned char *src, unsigned char *dest, float frequency, int width, int height){
    assert((src != nullptr) && (dest!=nullptr));
    //checkDimensionsEqualOrThrow(src, dest);

    int threadCount = threadIdx.x + (blockIdx.x * blockDim.x);

    for (;threadCount < height * width; threadCount += blockDim.x * gridDim.x)
    {
	int temp = 4 * threadCount;
        dest[temp] = 0;
        dest[temp+1] = 0;
        dest[temp+2] = 0;
        dest[temp+3] = 0;

        int y = threadCount / width;
	int x = threadCount % width;	
        float nx = -1.0f + (2.0f * x) / width;
        float ny = -1.0f + (2.0f * y) / height;

        // Calculate distance to center
        auto dist = std::sqrt(std::pow(ny, 2) + std::pow(nx, 2));

        // Calculate angle
        float angle = std::atan2(ny, nx);

        // Use a funky formula to make a lensing effect.
        auto src_dist = std::pow(std::sin(dist * M_PI / 2.0 * frequency), 2);

        // Check if this pixel lies within the source range, otherwise make this pixel transparent.
        if ((src_dist > 1.0f)) {
          continue;
        }

        // Calculate normalized lensed X and Y
        auto nsx = src_dist * std::cos(angle);
        auto nsy = src_dist * std::sin(angle);

        // Rescale to image size
        auto sx = int((nsx + 1.0) / 2 * width);
        auto sy = int((nsy + 1.0) / 2 * height);

        // Check bounds on source pixel
        if ((sx >= width) || (sy >= height)) {
          continue;
        }

        // Set destination pixel from source
        //dest->pixel(x, y) = src->pixel(sx, sy);
	int threadCountUpdated = sy*width+sx;
	int temp1 = 4*threadCountUpdated; 
        dest[temp]   = src[temp1];
        dest[temp+1] = src[temp1+1];
        dest[temp+2] = src[temp1+2];
        dest[temp+3] = src[temp1+3];
    }
}*/

__global__ void applyRippleCuda(unsigned char *src, unsigned char *dest, float frequency, int width, int height){
    //assert((src != nullptr) && (dest!=nullptr));
    //checkDimensionsEqualOrThrow(src, dest);

    //Thread Identifier for elements of width
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Thread Identifier for elements of height
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    //Check no thread goes out of bounds
    if ((x >= width) || (y >= height)) {
        return;
    }

    dest[(y * width + x) * 4 + blockIdx.z] = 0;

    float nx = -1.0f + (2.0f * x) / width;
    float ny = -1.0f + (2.0f * y) / height;

    // Calculate distance to center
    auto dist = std::sqrt(std::pow(ny, 2) + std::pow(nx, 2));

    // Calculate angle
    float angle = std::atan2(ny, nx);

    // Use a funky formula to make a lensing effect.
    auto src_dist = std::pow(std::sin(dist * M_PI / 2.0 * frequency), 2);

    // Check if this pixel lies within the source range, otherwise make this pixel transparent.
    if ((src_dist > 1.0f)) {
      return;
    }

    // Calculate normalized lensed X and Y
    auto nsx = src_dist * std::cos(angle);
    auto nsy = src_dist * std::sin(angle);

    // Rescale to image size
    auto sx = int((nsx + 1.0) / 2 * width);
    auto sy = int((nsy + 1.0) / 2 * height);

    // Check bounds on source pixel
    if ((sx >= width) || (sy >= height)) {
      return;
    }

    // Set destination pixel from source
    //dest->pixel(x, y) = src->pixel(sx, sy);
    dest[(y * width + x) * 4 + blockIdx.z] = src[(sy * width + sx) * 4 + blockIdx.z];
}



__global__ void convolutionKernelCuda(unsigned int height, unsigned int width,
                       int kernel_height, int kernel_width,
                       float kernel_scale, float *kernel_weights,
                       int kernel_xoff, int kernel_yoff,
                       unsigned char *source, unsigned char *dest) {
    //To distinguish between each threads and to get the number of thread that is being executed
    //Thread Identifier for elements of width
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Thread Identifier for elements of height
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    //Check no thread goes out of bounds
    if ((i >= width) || (j >= height)) {
        return;
    }

    // Convolution result
    auto c = 0.0;
    // Loop over every kernel weight
    for (int ky = -kernel_height / 2; ky <= kernel_height / 2; ky++) {
        for (int kx = -kernel_width / 2; kx <= kernel_width / 2; kx++) {
            // Convolute pixel x
            int cx = i + kx;
            // Convolute pixel y
            int cy = j + ky;
            // Bounds checking
            if ((cx >= 0) && (cy >= 0) && (cx < width) && (cy < height)) {
                // Pixel value
                auto v = (float) source[(cy * width + cx) * 4 + blockIdx.z];
                // Kernel weight
                auto k = kernel_weights[((ky + kernel_yoff) * kernel_width) + (kx + kernel_xoff)];
                // Multiply and accumulate
                c += v * k;
            }
        }
    }

    // Set the channel to the new color
    dest[(j * width + i) * 4 + blockIdx.z] = (unsigned char) (c * kernel_scale);
}


