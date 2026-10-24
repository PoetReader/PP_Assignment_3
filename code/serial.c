#include <stdio.h>
#include <stdlib.h>

int findMax(int *arr, size_t arr_size)
{
    int max = arr[0];
    for (int i = 1; i < arr_size; i++)
    {
        int current = arr[i];
        if (max < current)
        {
            max = current;
        }
    }
    return max;
}

int main(int argc, char const *argv[])
{
    int arr[] = {1, 2, 5, 7, 2, 4, 9, 10, 5, 3};
    size_t length_1 = 402653184;
    int *arr_1 = malloc(length_1 * sizeof(int));

    srand(29); // set the seed
    for (size_t i = 0; i < length_1; i++)
    {
        arr_1[i] = rand();
    }

    size_t size_1 = sizeof(arr_1) / sizeof(arr_1[0]);

    // Timer start arr 1
    int max_1 = findMax(arr_1, size_1);
    // Timer end arr 1

    printf("Array 1: Highest number: %d\n", max_1);

    return 0;
}
