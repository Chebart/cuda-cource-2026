#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Vector4 { float x, y, z, w; };

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

__global__ void scale_x_aos(const Vector4* vectors, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2.0f * vectors[i].x;
}

__global__ void scale_x_soa(const float* x, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2.0f * x[i];
}

void launch(bool soa, int n, const Vector4* vectors, const float* x, float* output) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    if (soa) scale_x_soa<<<blocks, threads>>>(x, output, n);
    else     scale_x_aos<<<blocks, threads>>>(vectors, output, n);
}

int main() {
    // Подготовить два расположения одних и тех же x.
    constexpr int n = 4 * 1024 * 1024;
    constexpr size_t bytes = n * sizeof(float);
    std::vector<Vector4> vectors(n);
    std::vector<float> x(n), result(n);

    for (int i = 0; i < n; ++i) {
        x[i] = float(i % 4096 + 1);
        vectors[i] = {x[i], -1.0f, -2.0f, -3.0f};
    }

    // Загрузить входы до замера.
    Vector4* d_vectors = nullptr;
    float *d_x = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_vectors, n * sizeof(Vector4)));
    check(cudaMalloc(&d_x, bytes));
    check(cudaMalloc(&d_out, bytes));

    check(cudaMemcpy(d_vectors, vectors.data(), n * sizeof(Vector4),
                     cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));

    // Прогреть обе версии и создать метки времени.
    for (int i = 0; i < 12; ++i) {
        launch(false, n, d_vectors, d_x, d_out);
        launch(true, n, d_vectors, d_x, d_out);
    }
    check(cudaGetLastError());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    constexpr int launches = 100;
    bool all_correct = true;
    for (int soa = 0; soa < 2; ++soa) {
        check(cudaMemset(d_out, 0, bytes));
        check(cudaEventRecord(start));
        for (int i = 0; i < launches; ++i)
            launch(soa, n, d_vectors, d_x, d_out);

        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop));

        check(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool correct = true;
        for (int i = 0; i < n; ++i)
            if (result[i] != 2.0f * x[i]) correct = false;

        std::printf("%s: [%g, %g, %g, ..., %g]; %.2f us/launch; %s\n",
                    soa ? "SoA" : "AoS", result[0], result[1], result[2],
                    result[n - 1], ms * 1000 / launches, correct ? "PASS" : "FAIL");
        all_correct = all_correct && correct;
    }

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_out));
    check(cudaFree(d_x));
    check(cudaFree(d_vectors));
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
