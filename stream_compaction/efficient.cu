#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#define BLOCK_SIZE 256

        __global__ void kernUpSweep(int n, int d, int* data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = 1 << (d + 1);
            int numActiveThreads = n >> (d + 1);

            if (idx >= numActiveThreads) return;

            int pos = (idx + 1) * stride - 1;
            int offset = stride >> 1;
            data[pos] += data[pos - offset];
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = 1 << (d + 1);
            int numActiveThreads = n >> (d + 1);

            if (idx >= numActiveThreads) return;

            int pos = (idx + 1) * stride - 1;
            int offset = stride >> 1;
            int temp = data[pos - offset];
            data[pos - offset] = data[pos];
            data[pos] += temp;
        }

        __global__ void kernScanBlocksEfficient(int n, const int* input, int* output, int* blockSums) {
            extern __shared__ int sdata[];

            int tid = threadIdx.x;
            int blockStart = blockIdx.x * BLOCK_SIZE * 2;

            int idx1 = blockStart + tid;
            int idx2 = blockStart + tid + BLOCK_SIZE;

            sdata[tid] = (idx1 < n) ? input[idx1] : 0;
            sdata[tid + BLOCK_SIZE] = (idx2 < n) ? input[idx2] : 0;
            __syncthreads();

            // Up-sweep
            int offset = 1;
            for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
                __syncthreads();
                if (tid < d) {
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = offset * (2 * tid + 2) - 1;
                    sdata[bi] += sdata[ai];
                }
                offset *= 2;
            }

            if (tid == 0) {
                if (blockSums) {
                    blockSums[blockIdx.x] = sdata[2 * BLOCK_SIZE - 1];
                }
                sdata[2 * BLOCK_SIZE - 1] = 0;
            }
            __syncthreads();

            // Down-sweep 
            for (int d = 1; d < 2 * BLOCK_SIZE; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (tid < d) {
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = offset * (2 * tid + 2) - 1;
                    int t = sdata[ai];
                    sdata[ai] = sdata[bi];
                    sdata[bi] += t;
                }
            }
            __syncthreads();

            if (idx1 < n) output[idx1] = sdata[tid];
            if (idx2 < n) output[idx2] = sdata[tid + BLOCK_SIZE];
        }

        __global__ void kernAddBlockSums(int n, int* data, const int* blockSums) {
            int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
            int blockId = blockIdx.x;

            if (blockId > 0 && idx < n) {
                int blockSum = blockSums[blockId];
                data[idx] += blockSum;
                if (idx + blockDim.x < n) {
                    data[idx + blockDim.x] += blockSum;
                }
            }
        }

        // Recursive scan function for block sums using shared memory
        void scanBlockSums(int n, int* dev_data) {
            if (n <= 1) return;

            const int elementsPerBlock = BLOCK_SIZE * 2;

            if (n <= elementsPerBlock) {
                // Use shared memory scan for small arrays
                int powerOfTwo = 1;
                while (powerOfTwo < n) powerOfTwo <<= 1;

                int* dev_temp;
                cudaMalloc((void**)&dev_temp, powerOfTwo * sizeof(int));
                cudaMemset(dev_temp, 0, powerOfTwo * sizeof(int));
                cudaMemcpy(dev_temp, dev_data, n * sizeof(int), cudaMemcpyDeviceToDevice);

                kernScanBlocksEfficient << <1, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_temp, dev_temp, nullptr
                    );

                cudaMemcpy(dev_data, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(dev_temp);
            }
            else {
                // Use multi-level approach for large arrays
                int numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;
                int powerOfTwo = 1;
                while (powerOfTwo < n) powerOfTwo <<= 1;

                int* dev_temp;
                int* dev_nextLevelSums;
                cudaMalloc((void**)&dev_temp, powerOfTwo * sizeof(int));
                cudaMalloc((void**)&dev_nextLevelSums, numBlocks * sizeof(int));

                cudaMemset(dev_temp, 0, powerOfTwo * sizeof(int));
                cudaMemcpy(dev_temp, dev_data, n * sizeof(int), cudaMemcpyDeviceToDevice);

                // Scan blocks
                kernScanBlocksEfficient << <numBlocks, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_temp, dev_temp, dev_nextLevelSums
                    );

                // Recursively scan the next level sums
                scanBlockSums(numBlocks, dev_nextLevelSums);

                // Add block sums back
                kernAddBlockSums << <numBlocks, BLOCK_SIZE >> > (powerOfTwo, dev_temp, dev_nextLevelSums);

                cudaMemcpy(dev_data, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);

                cudaFree(dev_temp);
                cudaFree(dev_nextLevelSums);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int powerOfTwo = 1 << ilog2ceil(n);
            const int elementsPerBlock = BLOCK_SIZE * 2;

            int* dev_data;
            cudaMalloc((void**)&dev_data, powerOfTwo * sizeof(int));
            cudaMemset(dev_data, 0, powerOfTwo * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc/memcpy failed");

            timer().startGpuTimer();
            // TODO

            if (powerOfTwo <= elementsPerBlock) {
                // Small array: single block scan
                kernScanBlocksEfficient << <1, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_data, dev_data, nullptr
                    );
                checkCUDAError("kernScanBlocksEfficient failed");
            }
            else {
                // Large array: multi-block scan with optimized block sums
                int numBlocks = (powerOfTwo + elementsPerBlock - 1) / elementsPerBlock;

                int* dev_blockSums;
                cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));

                // Scan each block
                kernScanBlocksEfficient << <numBlocks, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_data, dev_data, dev_blockSums
                    );
                checkCUDAError("kernScanBlocksEfficient failed");

                // Scan block sums using optimized recursive approach
                if (numBlocks > 1) {
                    scanBlockSums(numBlocks, dev_blockSums);

                    // Add block sums back
                    kernAddBlockSums << <numBlocks, BLOCK_SIZE >> > (powerOfTwo, dev_data, dev_blockSums);
                    checkCUDAError("kernAddBlockSums failed");
                }

                cudaFree(dev_blockSums);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            int powerOfTwo = 1 << ilog2ceil(n);
            const int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int elementsPerBlock = BLOCK_SIZE * 2;

            int* dev_input, * dev_bools, * dev_indices, * dev_output;

            cudaMalloc((void**)&dev_input, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, powerOfTwo * sizeof(int));
            cudaMalloc((void**)&dev_output, n * sizeof(int));

            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO

            // Map to boolean
            StreamCompaction::Common::kernMapToBoolean << <gridSize, BLOCK_SIZE >> > (n, dev_bools, dev_input);

            // Copy bools to indices for scan
            cudaMemset(dev_indices, 0, powerOfTwo * sizeof(int));
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // Scan the boolean array using optimized approach
            if (powerOfTwo <= elementsPerBlock) {
                kernScanBlocksEfficient << <1, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_indices, dev_indices, nullptr
                    );
            }
            else {
                int numBlocks = (powerOfTwo + elementsPerBlock - 1) / elementsPerBlock;
                int* dev_blockSums;
                cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));

                kernScanBlocksEfficient << <numBlocks, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_indices, dev_indices, dev_blockSums
                    );

                if (numBlocks > 1) {
                    // Use optimized recursive scan for block sums
                    scanBlockSums(numBlocks, dev_blockSums);
                    kernAddBlockSums << <numBlocks, BLOCK_SIZE >> > (powerOfTwo, dev_indices, dev_blockSums);
                }

                cudaFree(dev_blockSums);
            }

            // Get count
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            // Scatter
            StreamCompaction::Common::kernScatter << <gridSize, BLOCK_SIZE >> > (
                n, dev_output, dev_input, dev_bools, dev_indices
                );

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_output, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_input);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_output);

            return count;
        }


        void StreamCompaction::Efficient::scanDevice(int n, int* dev_out, const int* dev_in) {
            using namespace StreamCompaction::Common;
            const int powerOfTwo = 1 << ilog2ceil(n);
            const int elementsPerBlock = BLOCK_SIZE * 2;


            int* dev_data;
            cudaMalloc((void**)&dev_data, powerOfTwo * sizeof(int));
            cudaMemset(dev_data, 0, powerOfTwo * sizeof(int));

            cudaMemcpy(dev_data, dev_in, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("scanDevice: memcpy in");

            if (powerOfTwo <= elementsPerBlock) {

                kernScanBlocksEfficient << <1, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_data, dev_data, nullptr
                    );
                checkCUDAError("scanDevice: single-block scan");
            }
            else {

                const int numBlocks = (powerOfTwo + elementsPerBlock - 1) / elementsPerBlock;
                int* dev_blockSums;
                cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));


                kernScanBlocksEfficient << <numBlocks, BLOCK_SIZE, elementsPerBlock * sizeof(int) >> > (
                    powerOfTwo, dev_data, dev_data, dev_blockSums
                    );
                checkCUDAError("scanDevice: per-block scan");

                if (numBlocks > 1) {

                    scanBlockSums(numBlocks, dev_blockSums);

                    kernAddBlockSums << <numBlocks, BLOCK_SIZE >> > (powerOfTwo, dev_data, dev_blockSums);
                    checkCUDAError("scanDevice: add block sums");
                }
                cudaFree(dev_blockSums);
            }

            cudaMemcpy(dev_out, dev_data, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(dev_data);
        }



    }
}