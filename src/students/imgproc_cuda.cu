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
