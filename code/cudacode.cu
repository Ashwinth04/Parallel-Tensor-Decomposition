#include <iostream>
#include <cmath>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda.h>

// #define N 10

using namespace std;

__global__ void vectorNorm(double *a, double *result, int size) //Can be used for calculating the frobenius norm as well
{
    __shared__ double cache[1024];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    double temp = 0.0;

    while(tid < size)
    {
        temp += a[tid] * a[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
    {
        atomicAdd(result, cache[0]);
    }
}

double frobeniusNormCUDA(double* d_tensor, int total_size) {

    double* d_result;
    double result = 0.0;

    cudaMalloc((void **)&d_result, sizeof(double));

    int blockSize = 1024;
    int gridSize = (total_size + blockSize - 1)/ blockSize;

    vectorNorm<<<gridSize, blockSize>>>(d_tensor, d_result, total_size);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

__global__ void matrixMultiply(double *A, double *B, double *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matrixMultiplyCUDA(double* d_A, double* d_B, double* d_C, int m, int k, int n) {

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x,
        (m + blockDim.y - 1) / blockDim.y
    );

    printf("Grid dimension: %d, Block dimension: %d",256, ((n + blockDim.x - 1) / blockDim.x) * ((m + blockDim.y - 1) / blockDim.y));
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

__global__ void transposeMatrix(double* input, double* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

__global__ void applyHouseholderToColumns(double* A, double* u, int m, int n, int k, int ldA) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= k && j < n) {
        // Calculate dot product u^T * A(:,j)
        double dot_product = 0.0;
        for (int i = k; i < m; ++i) {
            dot_product += u[i - k] * A[i * ldA + j];
        }

        // Apply transformation A = (I - 2uu^T) * A
        for (int i = k; i < m; ++i) {
            A[i * ldA + j] -= 2 * u[i - k] * dot_product;
        }
    }
}

__global__ void updateUMatrix(double* U, double* u, int m, int k, int ldU) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        // Calculate dot product U(i,:) * u
        double dot_product = 0.0;
        for (int j = k; j < m; ++j) {
            dot_product += U[i * ldU + j] * u[j - k];
        }

        // Update U row
        for (int j = k; j < m; ++j) {
            U[i * ldU + j] -= 2 * dot_product * u[j - k];
        }
    }
}


__global__ void applyHouseholderToRows(double* A, double* u, int m, int n, int k, int ldA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        // Calculate dot product A(i,:) * u
        double dot_product = 0.0;
        for (int j = k + 1; j < n; ++j) {
            dot_product += A[i * ldA + j] * u[j - (k + 1)];
        }

        // Apply transformation A = A * (I - 2uu^T)
        for (int j = k + 1; j < n; ++j) {
            A[i * ldA + j] -= 2 * dot_product * u[j - (k + 1)];
        }
    }
}

__global__ void updateVMatrix(double* V, double* u, int n, int k, int ldV) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Calculate dot product V(i,:) * u
        double dot_product = 0.0;
        for (int j = k + 1; j < n; ++j) {
            dot_product += V[i * ldV + j] * u[j - (k + 1)];
        }

        // Update V row
        for (int j = k + 1; j < n; ++j) {
            V[i * ldV + j] -= 2 * dot_product * u[j - (k + 1)];
        }
    }
}

__global__ void extractColumn(double* A, double* x, int m, int n, int col, int startRow, int ldA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m - startRow) {
        x[i] = A[(i + startRow) * ldA + col];
    }
}

__global__ void extractRow(double* A, double* x, int m, int n, int row, int startCol, int ldA) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n - startCol) {
        x[j] = A[row * ldA + (j + startCol)];
    }
}

__global__ void initIdentityMatrix(double* M, int dim, int ldM) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dim) {
        for (int j = 0; j < dim; ++j) {
            M[i * ldM + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}


void householderBidiagonalizationCUDA(double* d_A, double* d_U, double* d_V, int m, int n, int ldA, int ldU, int ldV) {

    double* d_u;
    double* d_x;
    cudaMalloc(&d_u, max(m, n) * sizeof(double));
    cudaMalloc(&d_x, max(m, n) * sizeof(double));

    double* h_u = new double[max(m, n)];
    double* h_x = new double[max(m, n)];


    initIdentityMatrix<<<(m + 255) / 256, 256>>>(d_U, m, ldU);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
    initIdentityMatrix<<<(n + 255) / 256, 256>>>(d_V, n, ldV);
    // error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(error));
    //     // Handle error
    // }

    // Define block size for kernels
    int blockSize = 256;

    for (int k = 0; k < min(m, n); ++k) {
        extractColumn<<<(m-k + 255) / 256, 256>>>(d_A, d_x, m, n, k, k, ldA);
        // error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     printf("CUDA Error: %s\n", cudaGetErrorString(error));
        //     // Handle error
        // }
        cudaMemcpy(h_x, d_x, (m-k) * sizeof(double), cudaMemcpyDeviceToHost);

        double norm_x = 0.0;
        for (int i = 0; i < m-k; ++i) {
            norm_x += h_x[i] * h_x[i];
        }
        norm_x = sqrt(norm_x);

        double alpha = (h_x[0] >= 0) ? -norm_x : norm_x;
        for (int i = 0; i < m-k; ++i) {
            h_u[i] = h_x[i];
        }
        h_u[0] -= alpha;

        double norm_u = 0.0;
        for (int i = 0; i < m-k; ++i) {
            norm_u += h_u[i] * h_u[i];
        }
        norm_u = sqrt(norm_u);

        if (norm_u > 1e-5) {
            for (int i = 0; i < m-k; ++i) {
                h_u[i] /= norm_u;
            }

            cudaMemcpy(d_u, h_u, (m-k) * sizeof(double), cudaMemcpyHostToDevice);

            int numBlocksCol = (n - k + blockSize - 1) / blockSize;
            applyHouseholderToColumns<<<numBlocksCol, blockSize>>>(d_A, d_u, m, n, k, ldA);
            // error = cudaGetLastError();
            // if (error != cudaSuccess) {
            //     printf("CUDA Error: %s\n", cudaGetErrorString(error));
            //     // Handle error
            // }
            int numBlocksU = (m + blockSize - 1) / blockSize;
            updateUMatrix<<<numBlocksU, blockSize>>>(d_U, d_u, m, k, ldU);
            // error = cudaGetLastError();
            // if (error != cudaSuccess) {
            //     printf("CUDA Error: %s\n", cudaGetErrorString(error));
            //     // Handle error
            // }
        }

        if (k < n - 1) {
            extractRow<<<(n-k-1 + 255) / 256, 256>>>(d_A, d_x, m, n, k, k+1, ldA);

            cudaMemcpy(h_x, d_x, (n-k-1) * sizeof(double), cudaMemcpyDeviceToHost);

            norm_x = 0.0;
            for (int i = 0; i < n-k-1; ++i) {
                norm_x += h_x[i] * h_x[i];
            }
            norm_x = sqrt(norm_x);

            alpha = (h_x[0] >= 0) ? -norm_x : norm_x;
            for (int i = 0; i < n-k-1; ++i) {
                h_u[i] = h_x[i];
            }
            h_u[0] -= alpha;

            norm_u = 0.0;
            for (int i = 0; i < n-k-1; ++i) {
                norm_u += h_u[i] * h_u[i];
            }
            norm_u = sqrt(norm_u);

            if (norm_u > 1e-5) {
                for (int i = 0; i < n-k-1; ++i) {
                    h_u[i] /= norm_u;
                }

                cudaMemcpy(d_u, h_u, (n-k-1) * sizeof(double), cudaMemcpyHostToDevice);

                int numBlocksRow = (m + blockSize - 1) / blockSize;
                applyHouseholderToRows<<<numBlocksRow, blockSize>>>(d_A, d_u, m, n, k, ldA);

                int numBlocksV = (n + blockSize - 1) / blockSize;
                updateVMatrix<<<numBlocksV, blockSize>>>(d_V, d_u, n, k, ldV);
            }
        }
    }

    delete[] h_u;
    delete[] h_x;
    cudaFree(d_u);
    cudaFree(d_x);
}

__global__ void extractSingularValuesKernel(double* B, double* Sigma, int m, int n, int ldB, int ldSigma, int min_mn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < min_mn) {
        Sigma[i * ldSigma + i] = fabs(B[i * ldB + i]);
    }
}

__global__ void initZerosKernel(double* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] = 0.0;
    }
}

void extractSingularValuesCUDA(double* d_B, double* d_Sigma, int m, int n, int ldB, int ldSigma) {
    int min_mn = (m < n) ? m : n;
    
    int blockSize = 256;
    int numBlocks = (min_mn * min_mn + blockSize - 1) / blockSize;
    initZerosKernel<<<numBlocks, blockSize>>>(d_Sigma, min_mn * min_mn);

    numBlocks = (min_mn + blockSize - 1) / blockSize;
    extractSingularValuesKernel<<<numBlocks, blockSize>>>(d_B, d_Sigma, m, n, ldB, ldSigma, min_mn);
}

void reshapeTensorCUDA(double* d_input, double* d_output, int mode, int N, int total_size) {

    cudaMemcpy(d_output, d_input, total_size * sizeof(double), cudaMemcpyDeviceToDevice);
}

__global__ void copyColumn(double* src, double* dst, int m_src, int m_dst, 
                          int src_col, int dst_col, int ld_src, int ld_dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m_src) {
        dst[i * ld_dst + dst_col] = src[i * ld_src + src_col];
    }
}

__global__ void copyRow(double* src, double* dst, int n_src, int n_dst, 
                       int src_row, int dst_row, int ld_src, int ld_dst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < n_src) {
        dst[dst_row * ld_dst + j] = src[src_row * ld_src + j];
    }
}

__global__ void setDiagonalElement(double* matrix, double value, int ld, int idx) {
    matrix[idx * ld + idx] = value;
}

void deltaTruncatedSVDCUDA(double* d_A, double* d_U, double* d_S, double* d_V, 
                          double* d_U_trunc, double* d_S_trunc, double* d_V_trunc,
                          int m, int n, double delta, int* rank) {
    // Step 1: Compute SVD using Householder Bidiagonalization
    // Create temporary matrices
    double* d_B;
    cudaMalloc(&d_B, m * n * sizeof(double));
    cudaMemcpy(d_B, d_A, m * n * sizeof(double), cudaMemcpyDeviceToDevice);
  
    householderBidiagonalizationCUDA(d_B, d_U, d_V, m, n, n, m, n);
    printf("Bidiagonalization done\n\n\n");
    
    int min_mn = min(m, n);

    double* h_B_diag = new double[min_mn];
    double* h_S = new double[min_mn * min_mn];

    for (int i = 0; i < min_mn * min_mn; i++) {
        h_S[i] = 0.0;
    }

    // for (int i = 0; i < min_mn; i++) {
    //     cudaMemcpy(&h_B_diag[i], &d_B[i*n + i], sizeof(double), cudaMemcpyDeviceToHost);
    // }

    // for (int i = 0; i < min_mn; i++) {
    //     h_S[i*min_mn + i] = fabs(h_B_diag[i]);
    // }

    printf("Singular values extracted\n\n");
    
    printf("HII");
    
    // Step 2: Apply Delta Truncation
    // Copy singular values to host
    
    int k = 0;
 
    // for (int i = 0; i < min(m, n); ++i) {
    //     printf("h_S[%d] = %f, delta = %f\n", i, h_S[i], delta);
    //     if (h_S[i] > delta) {
    //         printf("Value exceeds delta, incrementing k\n");
    //         k++;
    //     }
    //     printf("Hello");
    // }
    // for (int i = 0; i < min(m, n); ++i) {
    //     printf("i = %d\n", i);
    // }

    printf("HII");
    *rank = k;
    printf("Rank: %d", *rank);

    if (k > 0) {

        int* h_indices = new int[k];
        
        int idx = 0;
        for (int i = 0; i < min(m, n); ++i) {
            if (h_S[i] > delta) {
                h_indices[idx++] = i;
            }
        }
        
        int* d_indices;
        cudaMalloc(&d_indices, k * sizeof(int));
        cudaMemcpy(d_indices, h_indices, k * sizeof(int), cudaMemcpyHostToDevice);
        
        for (int i = 0; i < k; ++i) {
            int col = h_indices[i];
            copyColumn<<<(m + 255) / 256, 256>>>(d_U, d_U_trunc, m, m, col, i, m, m);
            cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
            
        }
        
        for (int i = 0; i < k; ++i) {
            int row = h_indices[i];
            copyRow<<<(n + 255) / 256, 256>>>(d_V, d_V_trunc, n, n, row, i, n, k);
            
        }

        cudaMemset(d_S_trunc, 0, k * k * sizeof(double));
        for (int i = 0; i < k; ++i) {
            int idx = h_indices[i];
            setDiagonalElement<<<1, 1>>>(d_S_trunc, h_S[idx], k, i);
            
        }
        

        delete[] h_indices;
        cudaFree(d_indices);
    }

    delete[] h_S;
    cudaFree(d_B);
}


int main() {

    int m = 8;
    int n = 8;
    

    double* h_A = new double[m * n];

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < m * n; i++) {
        h_A[i] = dist(gen);
    }
    

    double *d_A, *d_U, *d_S, *d_V;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_U, m * m * sizeof(double));
    cudaMalloc(&d_S, m * n * sizeof(double));
    cudaMalloc(&d_V, n * n * sizeof(double));
    

    double *d_U_trunc, *d_S_trunc, *d_V_trunc;
    cudaMalloc(&d_U_trunc, m * m * sizeof(double));
    cudaMalloc(&d_S_trunc, m * n * sizeof(double));
    cudaMalloc(&d_V_trunc, n * n * sizeof(double));
    
    cudaMemcpy(d_A, h_A, m * n * sizeof(double), cudaMemcpyHostToDevice);

    double delta = 0.5;

    int rank;
    int* d_rank;
    cudaMalloc(&d_rank, sizeof(int));

    std::cout << "Running delta-truncated SVD..." << std::endl;
    deltaTruncatedSVDCUDA(d_A, d_U, d_S, d_V, d_U_trunc, d_S_trunc, d_V_trunc, m, n, delta, d_rank);
    
    cudaMemcpy(&rank, d_rank, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Truncated rank: " << rank << std::endl;
    
    double* h_S_trunc = new double[rank];
    cudaMemcpy(h_S_trunc, d_S_trunc, rank * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "Truncated singular values: ";
    for (int i = 0; i < rank; i++) {
        std::cout << h_S_trunc[i] << " ";
    }
    std::cout << std::endl;
    
    bool all_above_delta = true;
    for (int i = 0; i < rank; i++) {
        if (h_S_trunc[i] <= delta) {
            all_above_delta = false;
            break;
        }
    }
    std::cout << "All singular values above delta: " << (all_above_delta ? "Yes" : "No") << std::endl;
    
    // Clean up
    delete[] h_A;
    delete[] h_S_trunc;
    
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaFree(d_U_trunc);
    cudaFree(d_S_trunc);
    cudaFree(d_V_trunc);
    cudaFree(d_rank);
    
    return 0;
}

// int main() {
//     // Small 5D tensor (2x2x2x2x2)
//     const int N = 2;
//     const int total_size = N * N * N * N * N; // 32 elements
    
//     // Create and initialize tensor on host
//     double* h_tensor = new double[total_size];
//     mt19937 gen(42); // Fixed seed for reproducibility
//     normal_distribution<double> dist(0.0, 1.0);
    
//     for (int i = 0; i < total_size; i++) {
//         h_tensor[i] = dist(gen);
//     }
    
//     // Mode-1 unfolding: (2, 16)
//     int m = N;
//     int n = N * N * N * N;
    
//     // Print original matrix (first few elements)
//     cout << "Original matrix (first few elements):" << endl;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < min(5, n); j++) {
//             cout << fixed << setprecision(4) << h_tensor[i * n + j] << " ";
//         }
//         cout << endl;
//     }
    
//     // Make a copy of the original for verification
//     double* h_A_copy = new double[m * n];
//     memcpy(h_A_copy, h_tensor, m * n * sizeof(double));
    
//     // Allocate device memory
//     double *d_A, *d_U, *d_V;
//     cudaMalloc(&d_A, m * n * sizeof(double));
//     cudaMalloc(&d_U, m * m * sizeof(double));
//     cudaMalloc(&d_V, n * n * sizeof(double));
    
//     // Copy data to device
//     cudaMemcpy(d_A, h_tensor, m * n * sizeof(double), cudaMemcpyHostToDevice);
    
//     // Run bidiagonalization
//     householderBidiagonalizationCUDA(d_A, d_U, d_V, m, n, n, m, n);
    
//     // Copy results back
//     double* h_result = new double[m * n];
//     double* h_U = new double[m * m];
//     double* h_V = new double[n * n];
    
//     cudaMemcpy(h_result, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_U, d_U, m * m * sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_V, d_V, n * n * sizeof(double), cudaMemcpyDeviceToHost);
     
//     // Print bidiagonal matrix (should only have nonzeros on diagonal and superdiagonal)
//     cout << "\nBidiagonal matrix result:" << endl;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < min(5, n); j++) {
//             cout << fixed << setprecision(4) << h_result[i * n + j] << " ";
//         }
//         cout << endl;
//     }

//     int min_mn = min(m,n);
//     double *Sigma = new double[min_mn*min_mn];

//     for(int i=0;i<min_mn;i++)
//     {
//         for(int j=0;j<min_mn;j++)
//         {
//             Sigma[i*min_mn + j] = fabs(h_result[i*n + j]);
//         }
//     }

    
//     // Verify U is orthogonal by checking U*U^T
//     cout << "\nVerifying U is orthogonal (U*U^T):" << endl;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < m; j++) {
//             double sum = 0.0;
//             for (int k = 0; k < m; k++) {
//                 sum += h_U[i * m + k] * h_U[j * m + k];
//             }
//             cout << fixed << setprecision(4) << sum << " ";
//         }
//         cout << endl;
//     }
    
//     // Print part of U matrix
//     cout << "\nU matrix:" << endl;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < m; j++) {
//             cout << fixed << setprecision(4) << h_U[i * m + j] << " ";
//         }
//         cout << endl;
//     }
    
//     // Print part of V matrix (just first few columns)
//     cout << "\nV matrix (first few columns):" << endl;
//     for (int i = 0; i < min(3, n); i++) {
//         for (int j = 0; j < min(3, n); j++) {
//             cout << fixed << setprecision(4) << h_V[i * n + j] << " ";
//         }
//         cout << endl;
//     }
    
//     // Clean up
//     delete[] h_tensor;
//     delete[] h_A_copy;
//     delete[] h_result;
//     delete[] h_U;
//     delete[] h_V;
//     cudaFree(d_A);
//     cudaFree(d_U);
//     cudaFree(d_V);
    
//     return 0;
// }
