#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template<int WIDTH>
__global__ void read_column(const float* input, float* output) {
    __shared__ float tile[32][WIDTH];
    const int lane = threadIdx.x;
    const int base = blockIdx.x * 32 * 32;

    for (int row = 0; row < 32; ++row)
        tile[row][lane] = input[base + row * 32 + lane];

    __syncthreads();

    output[blockIdx.x * 32 + lane] = tile[lane][0];
}

void launch(int width, int tiles, const float* input, float* output) {
    if (width == 32) read_column<32><<<tiles, 32>>>(input, output);
    else             read_column<33><<<tiles, 32>>>(input, output);
}

int main(int argc, char** argv) {
    // Без аргумента измеряем 1024 тайла; аргумент 1 оставляет один.
    int tiles = 1024;
    if (argc == 2 && std::strcmp(argv[1], "1") == 0) tiles = 1;
    else if (argc != 1) return EXIT_FAILURE;

    std::vector<float> input(tiles * 32 * 32), output(tiles * 32);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(i);

    const size_t in_bytes = input.size() * sizeof(float);
    const size_t out_bytes = output.size() * sizeof(float);
    float *d_input = nullptr, *d_output = nullptr;
    check(cudaMalloc(&d_input, in_bytes));
    check(cudaMalloc(&d_output, out_bytes));
    check(cudaMemcpy(d_input, input.data(), in_bytes, cudaMemcpyHostToDevice));

    // Прогреть обе версии до замера.
    for (int i = 0; i < 12; ++i) {
        launch(32, tiles, d_input, d_output);
        launch(33, tiles, d_input, d_output);
    }
    check(cudaGetLastError());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    constexpr int launches = 100;
    bool all_correct = true;
    std::printf("Tiles: %d\n", tiles);

    for (int width = 32; width <= 33; ++width) {
        check(cudaMemset(d_output, 0xff, out_bytes));
        check(cudaEventRecord(start));
        for (int i = 0; i < launches; ++i)
            launch(width, tiles, d_input, d_output);

        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop));

        check(cudaMemcpy(output.data(), d_output, out_bytes, cudaMemcpyDeviceToHost));
        bool correct = true;
        for (int i = 0; i < tiles * 32; ++i)
            if (output[i] != input[i * 32]) correct = false;

        std::printf("WIDTH=%d: [%g, %g, %g, ..., %g]; %.2f us/launch; %s\n",
                    width, output[0], output[1], output[2], output[31],
                    ms * 1000 / launches, correct ? "PASS" : "FAIL");
        all_correct = all_correct && correct;
    }

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_output));
    check(cudaFree(d_input));
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
