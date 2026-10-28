#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <cuda.h> //Header file for CUDA
#include <time.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <unistd.h>

/**
 * @brief Helper function speedup
 * @param parallel time
 */
double speedUp(double serial, double parallel)
{
    return serial / parallel;
}

/**
 * @brief Helper function to get efficiency
 * @param speedup
 * @param units
 */
double efficiency(double speedup, int units)
{
    return speedup / units;
}

/**
 * @brief Helper function to find percent of time the parallel version spents compared to serial
 * @param serial_time
 * @param parallel_time
 */
double parallel_VS_serial(double serial, double parallel)
{
    return (parallel / serial) * 100;
}

/**
 * @brief Helper function to find min elapsed time
 * @param arr
 * @param testRuns
 */
double findMin(double arr[], int testRuns)
{

    double min = arr[0];
    for (int i = 1; i < testRuns; i++)
    {

        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    return min;
}

/**
 * @brief Helper to find mean of time arrays
 * @param arr
 * @param testRuns
 */
double findMean(double arr[], int testRuns)
{
    double sum = 0;
    for (int i = 0; i < testRuns; i++)
    {
        sum += arr[i];
    }

    return (double)sum / testRuns;
}

/**
 * @brief find Maximum used for serial version
 * @param arr
 * @param arr_size
 */
int findMax(int *arr, size_t arr_size)
{
    int max = arr[0];
    for (size_t i = 1; i < arr_size; i++)
    {
        int current = arr[i];
        if (max < current)
        {
            // printf("new maximum found at position %d between %d and %d\n", i--, max, current);
            max = current;
        }
    }
    return max;
}

/**
 * @brief find maximum on warop level
 * @param val
 */
__device__ int warpFindMax(int val)
{                                                   // regsiter value of the thread
    for (int offset = 16; offset > 0; offset >>= 1) // shift
    {
        int neighbour = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > neighbour) ? val : neighbour;
    }
    return val;
}

/**
 * @brief find max on the kernel
 * @param in //pointer to array on GPU mem
 * @param lenght // lenght of array
 * @param max // pointer to output int in GPU mem
 */
__global__ void kernelFindMax(const int *array, size_t length, int *max)
{
    __shared__ int warpMax[32]; // array to store the local max aand shared accros warps

    // grid-stride loop
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    int localMax = INT_MIN;

    for (size_t i = thread_id; i < length; i += stride)
    {
        if (array[i] > localMax)
        {
            localMax = array[i];
        }
    }
    // warp
    localMax = warpFindMax(localMax);

    // block
    int lane = threadIdx.x % 32;   // Position of thread inside its own warp
    int warpId = threadIdx.x / 32; // Number of warop inside its block

    if (lane == 0)
    {
        warpMax[warpId] = localMax; // only lane == 0 of each warp writes into the shared array
    }

    __syncthreads(); // as the shared memory will be read

    if (warpId == 0) // second time of reduction
    {
        // each thread in warp 0 loads their value into the shared array
        int val;
        if (threadIdx.x < blockDim.x / 32)
        {
            val = warpMax[threadIdx.x]; // load from shared array according to the amount of used warps per block
        }
        else
        {
            val = INT_MIN; // the rest is set to INT_MIN
        }

        val = warpFindMax(val); // second reduction

        if (threadIdx.x == 0)
        {
            atomicMax(max, val); // write result of block if greater than max
        }
    }
}

int main(int argc, char const *argv[])
{

    // Setup of the array to test on
    size_t length_1;

    size_t small = 134217728;
    size_t middle = 268435456;
    size_t large = 402653184;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: ./main {SMALL|MIDDLE|LARGE}\n");
        fprintf(stderr, "  SMALL  = %zu (0.5 GiB)\n", small);
        fprintf(stderr, "  MIDDLE = %zu (1.0 GiB)\n", middle);
        fprintf(stderr, "  LARGE  = %zu (1.5 GiB)\n", large);
        return EXIT_FAILURE;
    }

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) != 0)
    {
        fprintf(stderr, "gethostname failed\n");
        return EXIT_FAILURE;
    }

    char *dot = strchr(hostname, '.');
    if (dot)
        *dot = '\0'; // "c6-4.ifi.uit.no" -> "c6-4"

    char pathname[512];

    if (strcmp(argv[1], "SMALL") == 0)
    {
        length_1 = small;
        snprintf(pathname, sizeof(pathname), "results_%s_SMALL.txt", hostname);
    }
    else if (strcmp(argv[1], "MIDDLE") == 0)
    {
        length_1 = middle;
        snprintf(pathname, sizeof(pathname), "results_%s_MIDDLE.txt", hostname);
    }
    else if (strcmp(argv[1], "LARGE") == 0)
    {
        length_1 = large;
        snprintf(pathname, sizeof(pathname), "results_%s_LARGE.txt", hostname);
    }
    else
    {
        fprintf(stderr, "Unknown size: '%s'\n", argv[1]);
        return EXIT_FAILURE;
    }

    FILE *log = freopen(pathname, "w", stdout);
    if (log == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", pathname);
        return EXIT_FAILURE;
    }

    printf("Size is %zu\n", length_1);

    int *arr_1 = (int *)malloc(length_1 * sizeof(int));

    srand(52); // set the seed
    for (size_t i = 0; i < length_1; i++)
    {
        arr_1[i] = rand();
    }

    int init = INT_MIN;

    int testRuns = 30;

    double serial_times[testRuns];
    int max;

    // Measaurement arrays
    double parallel_times[testRuns];
    double parallel_host2device_times[testRuns];
    double parallel_device2host_times[testRuns];
    double parallel_kernel[testRuns];
    double pVSs[testRuns];
    double speed_Up[testRuns];
    double eff[testRuns];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int p = prop.multiProcessorCount;
    printf("SMs: %d\n", p);

    int *device_arr;
    int *device_max;

    cudaMalloc(&device_arr, length_1 * sizeof(int));
    cudaMalloc(&device_max, sizeof(int));

    int gpu_max;

    // for time measurements
    cudaEvent_t host2device_start, host2device_end, device2host_start, device2host_end, kernel_start, kernel_end;
    cudaEventCreate(&host2device_start);
    cudaEventCreate(&host2device_end);
    cudaEventCreate(&device2host_start);
    cudaEventCreate(&device2host_end);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    for (int current = 0; current < testRuns; current++)
    {
        /*---------------------Serial----------------------*/

        struct timespec start_serial, end_serial;
        clock_gettime(CLOCK_MONOTONIC, &start_serial);

        max = findMax(arr_1, length_1);

        clock_gettime(CLOCK_MONOTONIC, &end_serial);
        double serial_time = (end_serial.tv_sec - start_serial.tv_sec) * 1000.0 + (end_serial.tv_nsec - start_serial.tv_nsec) / 1e6;
        serial_times[current] = serial_time;

        /*--------------------Parallel---------------------*/

        cudaEventRecord(host2device_start); //---------------Timer Start-----------------
        cudaMemcpy(device_arr, arr_1, length_1 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_max, &init, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(host2device_end); //---------------Timer End-----------------

        cudaEventRecord(kernel_start); //---------------Timer Start-----------------
        kernelFindMax<<<1024, 256>>>(device_arr, length_1, device_max);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            // aufräumen, return EXIT_FAILURE
        }
        cudaEventRecord(kernel_end); //---------------Timer End-----------------

        cudaEventRecord(device2host_start); //---------------Timer Start-----------------
        cudaMemcpy(&gpu_max, device_max, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(device2host_end); //---------------Timer End-----------------

        cudaEventSynchronize(device2host_end);

        if (gpu_max != max)
        {
            fprintf(stderr, "something went wrong at run %d: parallel=%d serial=%d\n", current, gpu_max, max);

            cudaEventDestroy(host2device_start);
            cudaEventDestroy(host2device_end);
            cudaEventDestroy(device2host_start);
            cudaEventDestroy(device2host_end);
            cudaEventDestroy(kernel_start);
            cudaEventDestroy(kernel_end);

            cudaFree(device_max);
            cudaFree(device_arr);
            free(arr_1);
            return EXIT_FAILURE;
        }

        float time_host2device, time_device2host, time_kernel;
        cudaEventElapsedTime(&time_host2device, host2device_start, host2device_end);
        cudaEventElapsedTime(&time_device2host, device2host_start, device2host_end);
        cudaEventElapsedTime(&time_kernel, kernel_start, kernel_end);

        double time = time_host2device + time_device2host + time_kernel;
        parallel_times[current] = time;
        parallel_host2device_times[current] = time_host2device;
        parallel_device2host_times[current] = time_device2host;
        parallel_kernel[current] = time_kernel;
        pVSs[current] = parallel_VS_serial(serial_time, time);
        speed_Up[current] = speedUp(serial_time, time);
        eff[current] = efficiency(speed_Up[current], p) * 100;
    }

    printf("_____________________________________________________________________\n_____________________________________________________________________\nSerial max: %d\n", max);

    printf("_____________________________________________________________________\n_____________________________________________________________________\nParallel max: %d\n", gpu_max);
    printf("They are the same\n");

    for (int i = 0; i < testRuns; i++)
    {
        printf("-------------------------------------------------------------------\n%d. Run \nTotal Time elapsed: \nSerial: %f ms ---------------------- Parallel: %f ms\nPARALLEL TIMES --> Data copy from host to device: %f ms\nPARALLEL TIMES --> Data copy from device to host: %f ms\nPARALLEL TIMES --> Kernel: %f ms\nParallel VS Serial: %f %%\nSpeed Up: %fx\nEfficiency: %f %%\n", i + 1, serial_times[i], parallel_times[i], parallel_host2device_times[i], parallel_device2host_times[i], parallel_kernel[i], pVSs[i], speed_Up[i], eff[i]);
    }

    // print minimum
    double min_elapsed_time_serial = findMin(serial_times, testRuns);
    double min_elapsed_time_parallel = findMin(parallel_times, testRuns);
    double min_host2device_parallel = findMin(parallel_host2device_times, testRuns);
    double min_device2host_parallel = findMin(parallel_device2host_times, testRuns);
    double min_kernel = findMin(parallel_kernel, testRuns);
    double min_pVSs = parallel_VS_serial(min_elapsed_time_serial, min_elapsed_time_parallel);
    double min_speedUp = speedUp(min_elapsed_time_serial, min_elapsed_time_parallel);
    double min_efficiency = efficiency(min_speedUp, p) * 100;

    printf("_____________________________________________________________________\n_____________________________________________________________________\nMINIMUM TIME: \nSerial: %f ms ---------------------- Parallel: %f ms\nPARALLEL TIMES --> Data copy from host to device: %f ms\nPARALLEL TIMES --> Data copy from device to host: %f ms\nPARALLEL TIMES --> Kernel: %f ms\nParallel VS Serial: %f %%\nSpeed Up: %fx\nEfficiency: %f %%\n", min_elapsed_time_serial, min_elapsed_time_parallel, min_host2device_parallel, min_device2host_parallel, min_kernel, min_pVSs, min_speedUp, min_efficiency);

    // print mean
    double mean_serial = findMean(serial_times, testRuns);
    double mean_parallel = findMean(parallel_times, testRuns);
    double mean_host2device = findMean(parallel_host2device_times, testRuns);
    double mean_device2host = findMean(parallel_device2host_times, testRuns);
    double mean_kernel = findMean(parallel_kernel, testRuns);
    double mean_pVSs = parallel_VS_serial(mean_serial, mean_parallel);
    double mean_speedUp = speedUp(mean_serial, mean_parallel);
    double mean_efficiency = efficiency(mean_speedUp, p) * 100;

    printf("_____________________________________________________________________\n_____________________________________________________________________\nMEAN TIME: \nSerial: %f ms ---------------------- Parallel: %f ms\nPARALLEL TIMES --> Data copy from host to device: %f ms\nPARALLEL TIMES --> Data copy from device to host: %f ms\nPARALLEL TIMES --> Kernel: %f ms\nParallel VS Serial: %f %%\nSpeed Up: %fx\nEfficiency: %f %%\n", mean_serial, mean_parallel, mean_host2device, mean_device2host, mean_kernel, mean_pVSs, mean_speedUp, mean_efficiency);

    cudaEventDestroy(host2device_start);
    cudaEventDestroy(host2device_end);
    cudaEventDestroy(device2host_start);
    cudaEventDestroy(device2host_end);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);

    cudaFree(device_max);
    cudaFree(device_arr);
    free(arr_1);
    return 0;
}
