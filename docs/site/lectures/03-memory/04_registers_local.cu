#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

constexpr int values_per_thread = 64;

__global__ void add_steps(const float* input, float* output, int steps) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float values[values_per_thread];

    #pragma unroll
    for (int i = 0; i < values_per_thread; ++i)
        values[i] = input[i * stride + tid];

    #pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        #pragma unroll
        for (int i = 0; i < values_per_thread; ++i)
            values[i] += 1.0f;
    }

    #pragma unroll
    for (int i = 0; i < values_per_thread; ++i)
        output[i * stride + tid] = values[i];
}

int main() {
    // Подготовить одинаковый вход для обеих сборок.
    constexpr int blocks = 64, threads = 128, steps = 128;
    constexpr int n = blocks * threads * values_per_thread;
    constexpr size_t bytes = n * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = float(i % 17);

    float *d_input = nullptr, *d_output = nullptr;
    check(cudaMalloc(&d_input, bytes));
    check(cudaMalloc(&d_output, bytes));
    check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    // Прогреть GPU до замера.
    for (int i = 0; i < 12; ++i)
        add_steps<<<blocks, threads>>>(d_input, d_output, steps);
    check(cudaGetLastError());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    // Замерить серию одинаковых запусков на GPU.
    constexpr int launches = 100;
    check(cudaEventRecord(start));
    for (int i = 0; i < launches; ++i)
        add_steps<<<blocks, threads>>>(d_input, d_output, steps);
    check(cudaGetLastError());

    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop));

    // Проверить результат вне замера.
    check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (output[i] != input[i] + steps) correct = false;

    std::printf("First four: [%g, %g, %g, %g]\n",
                output[0], output[1], output[2], output[3]);
    std::printf("%s: checked %d values; %.2f us/launch\n",
                correct ? "PASS" : "FAIL", n, ms * 1000 / launches);

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_output));
    check(cudaFree(d_input));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
