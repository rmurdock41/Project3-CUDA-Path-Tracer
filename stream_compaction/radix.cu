#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "common.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Combined kernel: extract bit and flip in one pass
         */
        __global__ void kernExtractAndFlip(int n, int* bits, int* zeros, const int* data, int bit) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            int b = (data[i] >> bit) & 1;
            bits[i] = b;
            zeros[i] = 1 - b;
        }

        /**
         * Compute total zeros on GPU
         */
        __global__ void kernComputeTotal(int* totalZeros, const int* scan, const int* zeros, int n) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                *totalZeros = scan[n - 1] + zeros[n - 1];
            }
        }

        /**
         * Scatter elements directly using scan results
         */
        __global__ void kernScatterOptimized(int n, int* out, const int* in,
            const int* bits, const int* scan, int totalZeros) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            int pos = bits[i] ? (totalZeros + i - scan[i]) : scan[i];
            out[pos] = in[i];
        }

        /**
         * Find max value for bit optimization
         */
        __global__ void kernReduceMax(int n, const int* data, int* result) {
            __shared__ int sdata[256];

            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            sdata[tid] = (i < n) ? data[i] : 0;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && i + s < n) {
                    sdata[tid] = max(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                atomicMax(result, sdata[0]);
            }
        }

        /**
         * Count significant bits
         */
        __host__ int getBitCount(int x) {
            if (x <= 0) return 1;
            int count = 0;
            while (x > 0) {
                count++;
                x >>= 1;
            }
            return count;
        }

        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) return;

            // Allocate device memory
            int* dev_input;
            int* dev_output;
            int* dev_bits;
            int* dev_zeros;
            int* dev_scan;
            int* dev_totalZeros;
            int* dev_maxVal;

            cudaMalloc(&dev_input, n * sizeof(int));
            cudaMalloc(&dev_output, n * sizeof(int));
            cudaMalloc(&dev_bits, n * sizeof(int));
            cudaMalloc(&dev_zeros, n * sizeof(int));
            cudaMalloc(&dev_scan, n * sizeof(int));
            cudaMalloc(&dev_totalZeros, sizeof(int));
            cudaMalloc(&dev_maxVal, sizeof(int));
            checkCUDAError("cudaMalloc failed");

            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            // Grid configuration
            const int blockSize = 256;
            dim3 blockNum((n + blockSize - 1) / blockSize);

            // Find maximum value to determine bit count
            cudaMemset(dev_maxVal, 0, sizeof(int));
            kernReduceMax << <blockNum, blockSize >> > (n, dev_input, dev_maxVal);
            cudaDeviceSynchronize();

            int maxVal;
            cudaMemcpy(&maxVal, dev_maxVal, sizeof(int), cudaMemcpyDeviceToHost);
            int numBits = getBitCount(maxVal);

            // Create thrust device pointers once
            thrust::device_ptr<int> thrust_zeros = thrust::device_pointer_cast(dev_zeros);
            thrust::device_ptr<int> thrust_scan = thrust::device_pointer_cast(dev_scan);

            timer().startGpuTimer();

            // Process only necessary bits
            for (int bit = 0; bit < numBits; ++bit) {
                // Combined extract and flip
                kernExtractAndFlip << <blockNum, blockSize >> > (n, dev_bits, dev_zeros, dev_input, bit);

                // Thrust scan (minimal overhead)
                thrust::exclusive_scan(thrust_zeros, thrust_zeros + n, thrust_scan);

                // Compute total on GPU
                kernComputeTotal << <1, 1 >> > (dev_totalZeros, dev_scan, dev_zeros, n);

                // Get total zeros
                int totalZeros;
                cudaMemcpy(&totalZeros, dev_totalZeros, sizeof(int), cudaMemcpyDeviceToHost);

                // Scatter
                kernScatterOptimized << <blockNum, blockSize >> > (n, dev_output, dev_input,
                    dev_bits, dev_scan, totalZeros);

                // Swap buffers
                int* temp = dev_input;
                dev_input = dev_output;
                dev_output = temp;
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_input, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Final copy failed");

            // Cleanup
            cudaFree(dev_input);
            cudaFree(dev_output);
            cudaFree(dev_bits);
            cudaFree(dev_zeros);
            cudaFree(dev_scan);
            cudaFree(dev_totalZeros);
            cudaFree(dev_maxVal);
        }

        bool isSorted(int n, const int* data) {
            for (int i = 1; i < n; i++) {
                if (data[i] < data[i - 1]) {
                    return false;
                }
            }
            return true;
        }
    }
}