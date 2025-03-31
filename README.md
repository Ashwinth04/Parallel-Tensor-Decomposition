# High-Dimensional Tensor Decomposition using TT-SVD

## Overview
This project focuses on the decomposition of high-dimensional tensors using the **TT-SVD (Tensor Train Singular Value Decomposition) algorithm**. The TT-SVD method efficiently approximates a given high-dimensional tensor in the **Tensor Train (TT) format**, reducing storage and computational complexity while maintaining accuracy.

Implementations include:
- **Serial (C++)**: A straightforward implementation of TT-SVD.
- **Parallelized (OpenMP)**: Utilizes OpenMP for multi-threaded execution.
- **GPU-accelerated (CUDA)**: Implements CUDA for efficient tensor decomposition on GPUs.

## TT-SVD Algorithm
TT-SVD is a compression method that decomposes a tensor into a sequence of smaller matrices and tensors, known as TT-cores. Given a d-dimensional tensor \( A \), TT-SVD finds an approximation \( B \) such that:

\[ \|A - B\|_F \leq \varepsilon \|A\|_F \]

where \( \varepsilon \) is the prescribed accuracy.

The decomposition process involves iterative reshaping and singular value decomposition (SVD), as described in the provided algorithm:
1. Compute a truncation parameter based on \( \varepsilon \).
2. Iteratively reshape the tensor and perform **truncated SVD**.
3. Store the resulting TT-cores.

## Householder Bidiagonalization
Instead of directly using standard SVD, **Householder Bidiagonalization** is employed to improve numerical stability and efficiency. This technique transforms a matrix into a bidiagonal form using a series of Householder reflections, leading to:
- **Lower computational cost**: Reduces the complexity of SVD computation.
- **Better stability**: Prevents numerical issues common in high-dimensional decompositions.

Householder bidiagonalization is particularly useful in parallelized and GPU-based implementations, where numerical stability and efficiency are crucial.

## Implementations
- **C++ (Serial Implementation)**: Implements TT-SVD using standard matrix operations.
- **OpenMP (Multi-threaded)**: Accelerates tensor decomposition using shared-memory parallelism.
- **CUDA (GPU-accelerated)**: Uses CUDA to leverage GPU cores for high-speed SVD and tensor operations.

## Usage
1. Clone the repository:
   ```sh
   git clone <repo_link>
   cd ttsvd_project
   ```
2. Build the project for different versions:
   - **Serial (C++)**:
     ```sh
     g++ -o ttsvd_serial ttsvd_serial.cpp -O2
     ./ttsvd_serial
     ```
   - **OpenMP**:
     ```sh
     g++ -o ttsvd_openmp ttsvd_openmp.cpp -fopenmp -O2
     ./ttsvd_openmp
     ```
   - **CUDA**:
     ```sh
     nvcc -o ttsvd_cuda ttsvd_cuda.cu -O2
     ./ttsvd_cuda
     ```

## References
- Oseledets, I. V. (2011). "Tensor-train decomposition". SIAM Journal on Scientific Computing.
- Golub, G. H., Van Loan, C. F. (1996). "Matrix Computations".

## License
This project is released under the MIT License.
