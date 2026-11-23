#include <cuda_runtime.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <limits>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

void print_pool_usage(cudaMemPool_t pool, const char* label) {
    std::uint64_t used = 0, reserved = 0;
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    std::printf("%s: used=%" PRIu64 " B, reserved=%" PRIu64 " B\n",
                label, used, reserved);
}

int main() {
    // Настроить пул для двух запросов.
    int device = 0;
    check(cudaGetDevice(&device));
    cudaMemPool_t pool;
    check(cudaDeviceGetMemPool(&pool, device));

    // Просим пул сохранять свободный резерв при ожидании.
    auto threshold = std::numeric_limits<std::uint64_t>::max();
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                  &threshold));

    // Дважды взять и вернуть буфер.
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    print_pool_usage(pool, "before first alloc");

    for (int pass = 1; pass <= 2; ++pass) {
        void* buffer = nullptr;
        std::printf("Pass %d\n", pass);
        check(cudaMallocAsync(&buffer, 1024 * 1024, stream));
        check(cudaStreamSynchronize(stream));
        print_pool_usage(pool, "after alloc");

        check(cudaFreeAsync(buffer, stream));
        check(cudaStreamSynchronize(stream));
        print_pool_usage(pool, "after free");
    }

    // Удалить очередь.
    check(cudaStreamDestroy(stream));
}
