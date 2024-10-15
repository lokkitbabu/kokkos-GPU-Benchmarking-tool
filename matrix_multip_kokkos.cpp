#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>

// Function to get the number of GPUs available - Can handle 1-8 H100 GPUs
int getNumGPUs() {
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    return numGPUs;
}

// Matrix multiplication function with data partitioning across GPUs
void matrixMultiplyMultiGPU(const int N, const int numGPUs) {
    // Start the timer
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
            // Last GPU takes the remainder rows
            endRow += remainder;
        }
        int numRows = endRow - startRow;

        // Create execution space for this GPU
        Kokkos::Cuda cuda_instance;

        // Create subviews for this GPU's portion of the matrices
        Kokkos::View<double**, Kokkos::CudaSpace> A("A", numRows, N);
        Kokkos::View<double**, Kokkos::CudaSpace> B("B", N, N);
        Kokkos::View<double**, Kokkos::CudaSpace> C("C", numRows, N);

        // Initialize matrices A and B
        Kokkos::parallel_for("InitA", Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
            cuda_instance, {0, 0}, {numRows, N}),
            KOKKOS_LAMBDA(const int i, const int j) {
                A(i, j) = static_cast<double>(i + startRow + 1) * (j + 1);
        });

        Kokkos::parallel_for("InitB", Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
            cuda_instance, {0, 0}, {N, N}),
            KOKKOS_LAMBDA(const int i, const int j) {
                B(i, j) = static_cast<double>(i + 1) - (j + 1);
        });

        cuda_instance.fence();

        // Perform matrix multiplication C = A * B
        Kokkos::parallel_for("MatrixMultiply", Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
            cuda_instance, {0, 0}, {numRows, N}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
        });

        cuda_instance.fence();

        // Optionally, copy results back to host or to a global matrix C
        // For simplicity, we skip that step here

        #pragma omp critical
        {
            std::cout << "GPU " << gpu << " completed computation on rows " << startRow << " to " << endRow - 1 << std::endl;
        }
    }

    // Measure time taken for matrix multiplication
    double elapsedTime = timer.seconds();
    std::cout << "Time taken for matrix multiplication with size " << N << "x" << N << " using " << numGPUs << " GPUs: " << elapsedTime << " seconds." << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    int numGPUs = getNumGPUs();
    std::cout << "Number of GPUs available: " << numGPUs << std::endl;

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <start> <end> <step>" << std::endl;
        Kokkos::finalize();
        return 1;
    }

    // Parse command-line arguments
    int start = std::atoi(argv[1]);  // Starting matrix size
    int end = std::atoi(argv[2]);    // Ending matrix size
    int step = std::atoi(argv[3]);   // Step size

    // Loop over matrix sizes and perform multiplication
    for (int size = start; size <= end; size += step) {
        std::cout << "\nMatrix multiplication with size " << size << "x" << size << std::endl;
        matrixMultiplyMultiGPU(size, numGPUs);
        std::cout << "---------------------------------------" << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
