#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h> //Header file for CUDA

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
        //each thread in warp 0 loads their value into the shared array
        int val; 
        if (threadIdx.x < blockDim.x /32)
        {
            val = warpMax[threadIdx.x]; //load from shared array according to the amount of used warps per block
        } else {
            val = INT_MIN; // the rest is set to INT_MIN 
        }
        
        val = warpFindMax(val); // second reduction

        if (threadIdx.x == 0){
            *max = val; //write result
        }
    }
}

/**
 * int main(int argc, char∗ argv[])
 * {
 * int blk_ct;
 * //Number of thread blocks
 * int th_per_blk;
 * //Number of threads in each block
 * blk_ct = strtol(argv[1], NULL, 10);
 * //Get number of blocks from command line
 * th_per_blk = strtol(argv[2], NULL, 10);
 * //Get number of threads per block from command line
 * Hello<<<blk_ct, th_per_blk>>>();
 * Start blk_ct∗th_per_blk threads on GPU,
 * cudaDeviceSynchronize(); // Wait for GPU to finish
 * return 0;
}
 */

int main(int argc, char const *argv[])
{
    // test array
    int *arr_32 = (int *)malloc(32 * sizeof(int));

    // Setup of the array to test on
    size_t length_1 = 402653184;

    int *arr_1 = (int *)malloc(length_1 * sizeof(int));

    srand(29); // set the seed
    for (size_t i = 0; i < length_1; i++)
    {
        arr_1[i] = rand();
    }

    for (int i = 0; i < 32; i++)
    {
        arr_32[i] = rand();
    }

    /*---------------------Init GPU--------------------*/
    int *device_arr;
    int *device_max;
    cudaMalloc(&device_arr, length_1 * sizeof(int));
    cudaMalloc(&device_max, sizeof(int));
    cudaMemcpy(device_arr, arr_1, length_1 * sizeof(int), cudaMemcpyHostToDevice);
    int init = INT_MIN;
    cudaMemcpy(device_max, &init, sizeof(int), cudaMemcpyHostToDevice);
    /*--------------------Parallel---------------------*/

    /* -------- Kernel Start: 1 Block, 32 Threads -------- */
    kernelFindMax<<<1, 256>>>(device_arr, length_1, device_max);
    cudaDeviceSynchronize();

    /* -------- Copy res back to host -------- */
    int gpu_max;
    cudaMemcpy(&gpu_max, device_max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU max:    %d\n", gpu_max);

    /*---------------------Serial----------------------*/

    /*--------------------Timer start------------------*/
    int max = findMax(arr_1, length_1);
    /*--------------------Timer end--------------------*/

    printf("Highest number: %d\n", max);

    if (gpu_max == max)
    {
        printf("Yay, serial and parallel match\n");
    }
    else
    {
        printf("Something is wrong\n");
    }

    cudaFree(device_max);
    cudaFree(device_arr);
    free(arr_1);
    free(arr_32);
    return 0;
}
