#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

__global__ void sample(cudaTextureObject_t texture, float* output) {
    output[0] = tex2D<float>(texture, 1.0f, 1.0f);
}

int main() {
    // Загрузить четыре значения в CUDA-массив.
    const float pixels[4] = {0.0f, 4.0f, 8.0f, 12.0f};
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
    cudaArray_t array = nullptr;
    check(cudaMallocArray(&array, &channel, 2, 2));

    check(cudaMemcpy2DToArray(array, 0, 0, pixels, 2 * sizeof(float),
                              2 * sizeof(float), 2, cudaMemcpyHostToDevice));

    // Настроить линейное чтение.
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 0;  // Центры ячеек: 0.5 и 1.5.
    description.filterMode = cudaFilterModeLinear;

    cudaTextureObject_t texture = 0;
    check(cudaCreateTextureObject(&texture, &resource, &description, nullptr));
    float* output = nullptr;
    check(cudaMalloc(&output, sizeof(float)));

    // Прочитать значение в центре массива.
    sample<<<1, 1>>>(texture, output);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    float host = 0.0f;
    check(cudaMemcpy(&host, output, sizeof(host), cudaMemcpyDeviceToHost));

    constexpr float expected = 6.0f;
    bool correct = host == expected;
    std::printf("texture = %.1f, expected = %.1f\n", host, expected);
    std::puts(correct ? "PASS" : "FAIL");

    // Освободить объект и память.
    check(cudaFree(output));
    check(cudaDestroyTextureObject(texture));
    check(cudaFreeArray(array));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
