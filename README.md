# kokkos-GPU-Benchmarking-tool
# Kokkos GPU Benchmarking Tool

This repository contains a GPU benchmarking tool utilizing the [Kokkos](https://github.com/kokkos/kokkos) programming model. The tool is designed to evaluate the performance of various GPU architectures and provide insights into computational efficiency.

## Features

- Benchmarking different GPU architectures
- Performance evaluation using Kokkos
- Detailed performance metrics and analysis

## Requirements

- C++ compiler supporting C++11 or later
- [Kokkos](https://github.com/kokkos/kokkos) library installed
- CUDA-enabled GPU (for running GPU benchmarks)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/lokkitbabu/kokkos-GPU-Benchmarking-tool.git
    cd kokkos-GPU-Benchmarking-tool
    ```

2. Compile the project:
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ```

## Usage

1. Run the benchmarking tool:
    ```sh
    ./benchmark_tool
    ```

2. View the results and performance metrics output by the tool.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Kokkos](https://github.com/kokkos/kokkos) for providing the foundation for this benchmarking tool.
- The open-source community for their contributions and support.
