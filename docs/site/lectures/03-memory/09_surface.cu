#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

constexpr int W = 2, H = 2;

__global__ void writeSurface(cudaSurfaceObject_t surface) {
    int x = threadIdx.x, y = threadIdx.y;
    float value = 6.0f + y * W + x;
    surf2Dwrite(value, surface, x * sizeof(float), y); // X — байты, Y — строка.
}

int main() {
    // Подготовить CUDA-массив для записи.
    const size_t rowBytes = W * sizeof(float);
    cudaChannelFormatDesc format = cudaCreateChannelDesc<float>();
    cudaArray_t array = nullptr;
    check(cudaMallocArray(&array, &format, W, H, cudaArraySurfaceLoadStore));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    cudaSurfaceObject_t surface = 0;
    check(cudaCreateSurfaceObject(&surface, &resource));

    // Записать четыре числа прямо в массив.
    writeSurface<<<1, dim3(W, H)>>>(surface);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    // Прочитать и проверить результат на CPU.
    float host[W * H]{};
    check(cudaMemcpy2DFromArray(host, rowBytes, array, 0, 0,
                                rowBytes, H, cudaMemcpyDeviceToHost));

    const float expected[W * H] = {6.0f, 7.0f, 8.0f, 9.0f};
    bool correct = true;
    for (int i = 0; i < W * H; ++i) {
        correct = (host[i] == expected[i]) && correct;
    }

    std::printf("surface: %g %g; %g %g\n", host[0], host[1], host[2], host[3]);
    std::puts(correct ? "PASS" : "FAIL");

    // Освободить объект и массив.
    check(cudaDestroySurfaceObject(surface));
    check(cudaFreeArray(array));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
