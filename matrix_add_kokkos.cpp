#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <iomanip>

// Function to get the number of GPUs available
int getNumGPUs() {
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    return numGPUs;
}

// Matrix addition function with detailed output
double matrixAddMultiGPU(const int N, const int numGPUs) {
    Kokkos::Timer timer;

    // Use OpenMP to create threads for each GPU
    #pragma omp parallel num_threads(numGPUs)
    {
        int gpu = omp_get_thread_num(); // GPU ID corresponds to thread ID

        // Set the CUDA device for this thread
        cudaSetDevice(gpu);

        // Calculate the number of rows per GPU
        int rowsPerGPU = N / numGPUs;
        int remainder = N % numGPUs;

        // Calculate the start and end rows for this GPU
        int startRow = gpu * rowsPerGPU;
        int endRow = startRow + rowsPerGPU;
        if (gpu == numGPUs - 1) {
            endRow += remainder; // Add remaining rows to the last GPU
        }

        // Allocate matrices A, B, and C
        Kokkos::View<double**> A("A", N, N);
        Kokkos::View<double**> B("B", N, N);
        Kokkos::View<double**> C("C", N, N);

        // Initialize A and B
        Kokkos::parallel_for("Initialize A and B", Kokkos::RangePolicy<>(startRow, endRow), KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < N; ++j) {
                A(i, j) = i * N + j;
                B(i, j) = (i * N + j) * 0.5;
            }
        });

        // Perform matrix addition
        Kokkos::parallel_for("Matrix Addition", Kokkos::RangePolicy<>(startRow, endRow), KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) = A(i, j) + B(i, j);
            }
        });

        // Synchronize threads
        Kokkos::fence();

        // Output which GPU completed which rows
        #pragma omp critical
        {
            std::cout << "GPU " << gpu << " completed computation on rows " << startRow
                      << " to " << endRow - 1 << std::endl;
        }
    }

    // Return elapsed time
    return timer.seconds();
}

int main(int argc, char* argv[]) {
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    {
        int N = 3500; // Matrix size for this test
        int numGPUs = getNumGPUs();

        std::cout << "Matrix size: " << N << "x" << N << std::endl;
        std::cout << "Number of GPUs: " << numGPUs << std::endl;

        // Perform matrix addition
        double elapsed = matrixAddMultiGPU(N, numGPUs);

        std::cout << "Time taken for matrix addition with size " << N << "x" << N
                  << " using " << numGPUs << " GPUs: " << elapsed << " seconds." << std::endl;
    }

    // Finalize Kokkos
    Kokkos::finalize();
    return 0;
}