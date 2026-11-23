#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

__global__ void add_one(int* values) {
    int i = threadIdx.x;
    values[i] += 1;
}

int main() {
    // CPU заполняет массив в управляемой памяти.
    int* values = nullptr;
    check(cudaMallocManaged(&values, 4 * sizeof(int)));
    const int input[] = {2, 4, 6, 8};
    for (int i = 0; i < 4; ++i) values[i] = input[i];

    std::printf("before: [%d, %d, %d, %d]\n",
                values[0], values[1], values[2], values[3]);

    // GPU меняет тот же массив.
    add_one<<<1, 4>>>(values);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    // CPU читает массив после ожидания.
    std::printf("after: [%d, %d, %d, %d]\n",
                values[0], values[1], values[2], values[3]);
    bool correct = true;
    for (int i = 0; i < 4; ++i) {
        if (values[i] != input[i] + 1) correct = false;
    }

    check(cudaFree(values));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
