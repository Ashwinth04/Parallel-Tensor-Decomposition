#include <cuda_runtime.h>
#include <iostream>

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

__global__ void initZerosKernel(double* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] = 0.0;
    }
}

__global__ void copyColumn(double* src, double* dst, int m_src, int m_dst, int src_col, int dst_col, int ld_src, int ld_dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m_src) {
        dst[i * ld_dst + dst_col] = src[i * ld_src + src_col];
    }
}

__global__ void copyRow(double* src, double* dst, int n_src, int n_dst, int src_row, int dst_row, int ld_src, int ld_dst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < n_src) {
        dst[dst_row * ld_dst + j] = src[src_row * ld_src + j];
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

__global__ void setDiagonalElement(double* matrix, double value, int ld, int idx) {
    matrix[idx * ld + idx] = value;
}