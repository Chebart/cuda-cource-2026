#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

int main() {
    // Загрузить исходный массив на GPU.
    const int input[] = {2, 4, 6, 8};
    int* device = nullptr;
    check(cudaMalloc(&device, sizeof(input)));
    check(cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice));
    check(cudaDeviceSynchronize());

    // Подготовить приёмник и очередь команд.
    int* output = nullptr;
    cudaStream_t stream;
    check(cudaMallocHost(&output, sizeof(input)));
    check(cudaStreamCreate(&stream));

    // Поставить копию GPU → RAM в очередь.
    check(cudaMemcpyAsync(output, device, sizeof(input),
                          cudaMemcpyDeviceToHost, stream));
    // Здесь CPU может работать с другими данными.
    check(cudaStreamSynchronize(stream));

    // Прочитать результат после ожидания.
    std::printf("[%d, %d, %d, %d]\n", output[0], output[1], output[2], output[3]);
    bool correct = true;
    for (int i = 0; i < 4; ++i) {
        if (output[i] != input[i]) correct = false;
    }

    check(cudaStreamDestroy(stream));
    check(cudaFreeHost(output));
    check(cudaFree(device));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
