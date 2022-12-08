#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda.h>

__device__
void swap(int *a, int *b) {
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

__global__
void oddEvenSort_kernel(int* numbers_d, int n, int stride, bool odd) {
    int t_idx = (threadIdx.x + blockDim.x*blockIdx.x)*2;
    if (t_idx >= n-1) return;
    t_idx += odd ? 1 : 0;
    for (int j = t_idx; j < n-1; j += stride) {
        numbers_d[j] > numbers_d[j + 1] ? swap(&numbers_d[j], &numbers_d[j + 1]) : NULL;
    }
}

// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort
void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    int threads_per_block = 1024;
    int num_blocks = ceil(s/(threads_per_block*2.0));

    int* numbers_d;
    int stride = num_blocks*threads_per_block*2;
    auto s_bytes = s*sizeof(int);
    cudaMalloc((void**)&numbers_d, s_bytes);
    cudaMemcpy(numbers_d, &numbers[0], s_bytes, cudaMemcpyHostToDevice);
    for (int i = 0; i <= s; i++) {
        oddEvenSort_kernel<<<num_blocks, threads_per_block>>>(numbers_d, s, stride, (i+1)%2 == 1);
    }
    std::cout <<"Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaMemcpy(&numbers[0], numbers_d, s_bytes, cudaMemcpyDeviceToHost);
    cudaFree(numbers_d);
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main()
{
    constexpr unsigned int size = 524'288; // Number of elements in the input, 2^19

    // Initialize a vector with integers of value 0
    std::vector<int> numbers(size);
    // Populate our vector with (pseudo)random numbers
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    // print_sort_status(numbers);
    auto start = std::chrono::steady_clock::now();
    oddeven_sort(numbers);
    auto end = std::chrono::steady_clock::now();
    print_sort_status(numbers);
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
}