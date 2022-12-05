#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda.h>

__global__
void swap(int* a, int* b) {
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

__global__
void oddEvenSort_kernel(int* numbers_d, int n, int stride) {
    int t_idx = threadIdx.x + blockDim.x*blockId.x;

    if (t_idx*stride >= n)
    for (int i = 1; i <= s; i++) {
        int idx = t_idx*stride;
        // if (i % 2 == 1) idx += 1;
        idx += i % 2 == 1 ? 1 : 0;
        for (int j = idx; j < idx + stride; j = j + 2) {
            if (numbers_d[j] > numbers_d[j + 1]) {
                swap(&numbers_d[j], &numbers_d[j + 1]);
            }
        }
        __syncthreads()
    }
}

// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort
void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    int num_blocks = 1;
    int threads_per_block = 2048;

    int* numbers_d;
    int stride = (int)std::ceil(s/2048);
    auto s_bytes = s*sizeof(int)
    cudaMalloc((void**)&numbers_d, s_bytes);
    cudaMemcpy(numbers_d, numbers, s_bytes, cudaMemcpyHostToDevice);

    oddeven_sort_kernel<<<num_blocks, threads_per_block>>>(numbers_d, n, stride);

    cudaFree(numbers_d);
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main()
{
    constexpr unsigned int size = 100'000; // Number of elements in the input

    // Initialize a vector with integers of value 0
    std::vector<int> numbers(size);
    // Populate our vector with (pseudo)random numbers
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    print_sort_status(numbers);
    auto start = std::chrono::steady_clock::now();
    oddeven_sort(numbers);
    auto end = std::chrono::steady_clock::now();
    print_sort_status(numbers);
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
}