#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#define BLOCK_SIZE 256

        // TODO: __global__

        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            int step = 1 << d;  // 2^d

            if (index >= step) {
                odata[index] = idata[index] + idata[index - step];
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Allocate device memory (outside timing)
            int* dev_buf1, * dev_buf2;
            cudaMalloc((void**)&dev_buf1, n * sizeof(int));
            cudaMalloc((void**)&dev_buf2, n * sizeof(int));
            checkCUDAError("cudaMalloc failed");



            // Copy shifted data to first buffer (outside timing)
            cudaMemcpy(dev_buf1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");


            timer().startGpuTimer();
            // TODO

            // Calculate grid and block dimensions
            dim3 blockSize(BLOCK_SIZE);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

            // Perform scan iterations
            int logn = ilog2ceil(n);
            for (int d = 0; d < logn; d++) {
                kernNaiveScan << <gridSize, blockSize >> > (n, d, dev_buf2, dev_buf1);
                cudaDeviceSynchronize();
                checkCUDAError("kernNaiveScan failed");

                // Swap buffers for next iteration
                int* temp = dev_buf1;
                dev_buf1 = dev_buf2;
                dev_buf2 = temp;
            }


            // Convert inclusive to exclusive scan
            kernInclusiveToExclusive << <gridSize, blockSize >> > (n, dev_buf2, dev_buf1);
            cudaDeviceSynchronize();
            checkCUDAError("kernInclusiveToExclusive failed");
            timer().endGpuTimer();

            // Copy result back to host (outside timing)
            cudaMemcpy(odata, dev_buf2, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Final copy failed");

            // Clean up memory (outside timing)
            cudaFree(dev_buf1);
            cudaFree(dev_buf2);
        }
    }
}