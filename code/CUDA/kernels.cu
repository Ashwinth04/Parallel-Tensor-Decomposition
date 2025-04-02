#include <cuda_runtime.h>
#include <iostream>

// Kernel to compute vector or Frobenius norm
__global__ void vectorNorm(double *a, double *result, int size) {
    __shared__ double cache[1024];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    double temp = 0.0;


    while(tid < size) {
        temp += a[tid] * a[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(result, cache[0]);
    }
}

// Kernel for matrix multiplication (C = A * B)
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

// Kernel to transpose a matrix
__global__ void transposeMatrix(double* input, double* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

// Kernel to initialize matrix to zeros
__global__ void initZerosKernel(double* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] = 0.0;
    }
}

// Apply Householder reflection to matrix columns (right multiplication)
__global__ void applyHouseholderToColumns(double* A, double* u, int m, int n, int k, int ldA) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= k && j < n) {
        // Calculate dot product u^T * A(:,j) starting from row k
        double dot_product = 0.0;
        for (int i = 0; i < m-k; ++i) {
            dot_product += u[i] * A[(i+k) * ldA + j];
        }
        
        // Apply transformation A = (I - 2uu^T) * A
        for (int i = 0; i < m-k; ++i) {
            A[(i+k) * ldA + j] -= 2.0 * u[i] * dot_product;
        }
    }
}

// Update U matrix with the Householder transformation
__global__ void updateUMatrix(double* U, double* u, int m, int k, int ldU) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m) {
        // Calculate dot product U(i,:) * u for columns starting at k
        double dot_product = 0.0;
        for (int j = 0; j < m-k; ++j) {
            dot_product += U[i * ldU + (j+k)] * u[j];
        }
        
        // Update U row: U = U * (I - 2uu^T)
        for (int j = 0; j < m-k; ++j) {
            U[i * ldU + (j+k)] -= 2.0 * dot_product * u[j];
        }
    }
}

// Apply Householder reflection to matrix rows (left multiplication)
__global__ void applyHouseholderToRows(double* A, double* u, int m, int n, int k, int ldA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m) {
        // Calculate dot product A(i,:) * u for columns after k
        double dot_product = 0.0;
        for (int j = 0; j < n-(k+1); ++j) {
            dot_product += A[i * ldA + (j+k+1)] * u[j];
        }
        
        // Apply transformation A = A * (I - 2uu^T)
        for (int j = 0; j < n-(k+1); ++j) {
            A[i * ldA + (j+k+1)] -= 2.0 * dot_product * u[j];
        }
    }
}

// Update V matrix with the Householder transformation
__global__ void updateVMatrix(double* V, double* u, int n, int k, int ldV) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        // Calculate dot product V(i,:) * u for columns after k
        double dot_product = 0.0;
        for (int j = 0; j < n-(k+1); ++j) {
            dot_product += V[i * ldV + (j+k+1)] * u[j];
        }
        
        // Update V row: V = V * (I - 2uu^T)
        for (int j = 0; j < n-(k+1); ++j) {
            V[i * ldV + (j+k+1)] -= 2.0 * dot_product * u[j];
        }
    }
}

// Extract a column from a matrix
__global__ void extractColumn(double* A, double* x, int m, int n, int col, int startRow, int ldA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m - startRow) {
        // Extract A(startRow+i, col) to x[i]
        x[i] = A[(i + startRow) * ldA + col];
    }
}

// Extract a row from a matrix
__global__ void extractRow(double* A, double* x, int m, int n, int row, int startCol, int ldA) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < n - startCol) {
        // Extract A(row, startCol+j) to x[j]
        x[j] = A[row * ldA + (j + startCol)];
    }
}

// Copy a column from source to destination matrix
__global__ void copyColumn(double* src, double* dst, int m_src, int m_dst, int src_col, int dst_col, int ld_src, int ld_dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m_src && i < m_dst) {
        dst[i * ld_dst + dst_col] = src[i * ld_src + src_col];
    }
}

// Copy a row from source to destination matrix
__global__ void copyRow(double* src, double* dst, int n_src, int n_dst, int src_row, int dst_row, int ld_src, int ld_dst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < n_src && j < n_dst) {
        dst[dst_row * ld_dst + j] = src[src_row * ld_src + j];
    }
}

// Initialize a matrix as an identity matrix
__global__ void initIdentityMatrix(double* M, int dim, int ldM) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dim) {
        for (int j = 0; j < dim; ++j) {
            M[i * ldM + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// Set a specific diagonal element of a matrix
__global__ void setDiagonalElement(double* matrix, double value, int ld, int idx) {
    matrix[idx * ld + idx] = value;
}