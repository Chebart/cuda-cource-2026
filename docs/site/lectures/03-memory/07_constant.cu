#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__constant__ int table[32];

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

__global__ void sum_table(int* output, int shifted) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % 32;
    const int first = lane * shifted;

    int sum = 0;
    #pragma unroll 1
    for (int step = 0; step < 32; ++step)
        sum += table[(first + step) % 32];

    output[tid] = sum;
}

int main() {
    // Загрузить одну таблицу перед обоими запусками.
    int host_table[32];
    for (int i = 0; i < 32; ++i) host_table[i] = i + 1;
    check(cudaMemcpyToSymbol(table, host_table, sizeof(host_table)));

    constexpr int blocks = 256, threads = 128, n = blocks * threads;
    constexpr size_t bytes = n * sizeof(int);
    std::vector<int> result(n);
    int* device = nullptr;
    check(cudaMalloc(&device, bytes));

    // Прогреть оба порядка чтения.
    for (int i = 0; i < 12; ++i) {
        sum_table<<<blocks, threads>>>(device, 0);
        sum_table<<<blocks, threads>>>(device, 1);
    }
    check(cudaGetLastError());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    constexpr int launches = 100;
    bool all_correct = true;
    for (int shifted = 0; shifted < 2; ++shifted) {
        check(cudaMemset(device, 0, bytes));
        check(cudaEventRecord(start));
        for (int i = 0; i < launches; ++i)
            sum_table<<<blocks, threads>>>(device, shifted);

        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop));

        check(cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost));
        bool correct = true;
        for (int sum : result)
            if (sum != 528) correct = false;

        std::printf("%s: sum = %d; %.2f us/launch; %s\n",
                    shifted ? "Different addresses" : "One address", result[0],
                    ms * 1000 / launches, correct ? "PASS" : "FAIL");
        all_correct = all_correct && correct;
    }

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(device));
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
