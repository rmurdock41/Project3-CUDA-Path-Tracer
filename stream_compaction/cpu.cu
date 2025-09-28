#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // Exclusive prefix sum 
            if (n > 0) {
                odata[0] = 0;  // First element is always 0 for exclusive scan
                for (int i = 1; i < n; i++) {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int writeIndex = 0;

            // Simple loop: copy non-zero elements to output array
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[writeIndex] = idata[i];
                    writeIndex++;
                }
            }
            timer().endCpuTimer();
            return writeIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // Map to boolean array (0 or 1)
            int* boolArray = new int[n];
            for (int i = 0; i < n; i++) {
                boolArray[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Scan the boolean array to get indices
            int* scanArray = new int[n];
            if (n > 0) {
                scanArray[0] = 0;
                for (int i = 1; i < n; i++) {
                    scanArray[i] = scanArray[i - 1] + boolArray[i - 1];
                }
            }

            // Place elements at their computed positions
            int totalElements = 0;
            for (int i = 0; i < n; i++) {
                if (boolArray[i] == 1) {
                    odata[scanArray[i]] = idata[i];
                    totalElements = scanArray[i] + 1;  // Track the final count
                }
            }

            // Clean up temporary arrays
            delete[] boolArray;
            delete[] scanArray;

            timer().endCpuTimer();
            return totalElements;
        }
    }
}
