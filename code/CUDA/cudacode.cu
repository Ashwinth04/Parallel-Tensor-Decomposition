#include <iostream>
#include <cmath>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda.h>

#define EPSILON 1e-9 

using namespace std;
using namespace chrono;

// External kernel declarations
extern __global__ void vectorNorm(double *a, double *result, int size);
extern __global__ void matrixMultiply(double *A, double *B, double *C, int m, int k, int n);
extern __global__ void transposeMatrix(double* input, double* output, int rows, int cols);
extern __global__ void applyHouseholderToColumns(double* A, double* u, int m, int n, int k, int ldA);
extern __global__ void updateUMatrix(double* U, double* u, int m, int k, int ldU);
extern __global__ void applyHouseholderToRows(double* A, double* u, int m, int n, int k, int ldA);
extern __global__ void updateVMatrix(double* V, double* u, int n, int k, int ldV);
extern __global__ void initZerosKernel(double* matrix, int size);
extern __global__ void copyColumn(double* src, double* dst, int m_src, int m_dst, int src_col, int dst_col, int ld_src, int ld_dst);
extern __global__ void copyRow(double* src, double* dst, int n_src, int n_dst, int src_row, int dst_row, int ld_src, int ld_dst);
extern __global__ void extractColumn(double* A, double* x, int m, int n, int col, int startRow, int ldA);
extern __global__ void extractRow(double* A, double* x, int m, int n, int row, int startCol, int ldA);
extern __global__ void initIdentityMatrix(double* M, int dim, int ldM);
extern __global__ void setDiagonalElement(double* matrix, double value, int ld, int idx);

// Calculate Frobenius norm using CUDA
double frobeniusNormCUDA(double* d_tensor, int total_size) {
    double* d_result;
    double result = 0.0;

    cudaMalloc((void **)&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));  // Initialize to zero

    int blockSize = 1024;
    int gridSize = (total_size + blockSize - 1)/ blockSize;

    vectorNorm<<<gridSize, blockSize>>>(d_tensor, d_result, total_size);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return sqrt(result);  // Return the square root for actual norm
}

// Matrix multiplication with CUDA
void matrixMultiplyCUDA(double* d_A, double* d_B, double* d_C, int m, int k, int n) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x,
        (m + blockDim.y - 1) / blockDim.y
    );

    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in matrixMultiplyCUDA: %s\n", cudaGetErrorString(error));
    }
}

// Householder bidiagonalization
void householderBidiagonalizationCUDA(double* d_A, double* d_U, double* d_V, int m, int n, int ldA, int ldU, int ldV) {
    double* d_u;
    double* d_x;
    cudaMalloc(&d_u, max(m, n) * sizeof(double));
    cudaMalloc(&d_x, max(m, n) * sizeof(double));

    double* h_u = new double[max(m, n)];
    double* h_x = new double[max(m, n)];

    // Initialize U and V as identity matrices
    initIdentityMatrix<<<(m + 255) / 256, 256>>>(d_U, m, ldU);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in initIdentityMatrix(U): %s\n", cudaGetErrorString(error));
    }
    
    initIdentityMatrix<<<(n + 255) / 256, 256>>>(d_V, n, ldV);
    cudaDeviceSynchronize();
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in initIdentityMatrix(V): %s\n", cudaGetErrorString(error));
    }

    int blockSize = 256;

    // Main bidiagonalization loop
    for (int k = 0; k < min(m, n); ++k) {
        // Extract column for householder reflection
        extractColumn<<<(m-k + 255) / 256, 256>>>(d_A, d_x, m, n, k, k, ldA);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_x, d_x, (m-k) * sizeof(double), cudaMemcpyDeviceToHost);

        // Calculate norm of the column
        double norm_x = 0.0;
        for (int i = 0; i < m-k; ++i) {
            norm_x += h_x[i] * h_x[i];
        }
        norm_x = sqrt(norm_x);

        if (norm_x < 1e-10) continue;  // Skip if column is already zeroed

        // Compute alpha and build u vector
        double alpha = (h_x[0] >= 0) ? -norm_x : norm_x;
        for (int i = 0; i < m-k; ++i) {
            h_u[i] = h_x[i];
        }
        h_u[0] -= alpha;

        // Normalize u
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

            // Apply column-wise householder transformation
            int numBlocksCol = (n - k + blockSize - 1) / blockSize;
            applyHouseholderToColumns<<<numBlocksCol, blockSize>>>(d_A, d_u, m, n, k, ldA);
            cudaDeviceSynchronize();

            // Update U matrix
            int numBlocksU = (m + blockSize - 1) / blockSize;
            updateUMatrix<<<numBlocksU, blockSize>>>(d_U, d_u, m, k, ldU);
            cudaDeviceSynchronize();
        }

        // Process superdiagonal element if not the last column
        if (k < n - 1) {
            extractRow<<<(n-k-1 + 255) / 256, 256>>>(d_A, d_x, m, n, k, k+1, ldA);
            cudaDeviceSynchronize();
            
            cudaMemcpy(h_x, d_x, (n-k-1) * sizeof(double), cudaMemcpyDeviceToHost);

            norm_x = 0.0;
            for (int i = 0; i < n-k-1; ++i) {
                norm_x += h_x[i] * h_x[i];
            }
            norm_x = sqrt(norm_x);

            if (norm_x < 1e-10) continue;  // Skip if row is already zeroed

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

                // Apply row-wise householder transformation
                int numBlocksRow = (m + blockSize - 1) / blockSize;
                applyHouseholderToRows<<<numBlocksRow, blockSize>>>(d_A, d_u, m, n, k, ldA);
                cudaDeviceSynchronize();

                // Update V matrix
                int numBlocksV = (n + blockSize - 1) / blockSize;
                updateVMatrix<<<numBlocksV, blockSize>>>(d_V, d_u, n, k, ldV);
                cudaDeviceSynchronize();
            }
        }
    }

    delete[] h_u;
    delete[] h_x;
    cudaFree(d_u);
    cudaFree(d_x);
}

// Reshape tensor for next iteration (mode-n unfolding)
void reshapeTensorCUDA(double* d_input, double* d_output, int rows, int cols, int new_rows, int new_cols) {
    // For TT-SVD, we need to properly reshape the data for the next core computation
    cudaMemcpy(d_output, d_input, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
}

// Extract bidiagonal components 
void extract_bidiagonal(double *h_result, double *d, double *e, int m, int n) {
    for (int i = 0; i < min(m, n); i++) {
        d[i] = h_result[i * n + i]; // Main diagonal
        if (i < min(m, n) - 1) {
            e[i] = h_result[i * n + i + 1]; // Superdiagonal
        } else {
            e[i] = 0.0;
        }
    }
}

// Compute singular values from bidiagonal matrix
void compute_singular_values(double *d, double *e, int n, vector<double> &singular_values) {
    // This is a simplified SVD computation - in production code you'd use an
    // iterative method like QR iterations with shifts
    for (int i = 0; i < n; i++) {
        singular_values[i] = fabs(d[i]); // Initial guess: diagonal values
    }

    // Simple fixed-iteration algorithm for finding singular values
    // In practice, a more sophisticated algorithm would be used
    for (int iter = 0; iter < 30; iter++) {
        bool converged = true;
        for (int i = 0; i < n - 1; i++) {
            if (fabs(e[i]) > EPSILON * (fabs(d[i]) + fabs(d[i+1]))) {
                converged = false;
                
                // Simplified Givens rotation to eliminate superdiagonal element
                double t = singular_values[i] * singular_values[i] - singular_values[n-1] * singular_values[n-1];
                double s = sqrt(t * t + e[i] * e[i]);
                double c = t / s;
                double s_rot = e[i] / s;

                double temp = singular_values[i] * c + e[i] * s_rot;
                e[i] = -singular_values[i] * s_rot + e[i] * c;
                singular_values[i] = temp;
            }
        }
        
        if (converged) break;
    }
    
    // Sort singular values in descending order
    sort(singular_values.begin(), singular_values.end(), greater<double>());
}

// Get singular values from matrix
vector<double> get_singular_values(double *h_result, int m, int n) {
    int min_dim = min(m, n);
    double *d = new double[min_dim];
    double *e = new double[min_dim];
    vector<double> singular_values(min_dim);

    extract_bidiagonal(h_result, d, e, m, n);
    compute_singular_values(d, e, min_dim, singular_values);

    delete[] d;
    delete[] e;
    return singular_values;
}

// Perform SVD with delta truncation
int deltaTruncatedSVDCUDA(double* d_A, double* d_U, double* d_S, double* d_V, double* d_U_trunc, double* d_S_trunc, double* d_V_trunc, int m, int n, double delta) {
    // Perform bidiagonalization
    householderBidiagonalizationCUDA(d_A, d_U, d_V, m, n, n, m, n);
    
    // Get matrices to host for singular value computation
    double* h_result = new double[m * n];
    double* h_U = new double[m * m];
    double* h_V = new double[n * n];
    
    cudaMemcpy(h_result, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Get singular values
    vector<double> singular_values = get_singular_values(h_result, m, n);
    
    cout << "Computed singular values: ";
    for (int i = 0; i < min(5, (int)singular_values.size()); i++) {
        cout << singular_values[i] << " ";
    }
    cout << "..." << endl;
    
    // Count number of values above threshold
    int k = 0;
    for (double sv : singular_values) {
        if (sv > delta) k++;
    }
    
    cout << "Keeping " << k << " singular values above threshold " << delta << endl;

    // Handle truncation
    if (k > 0) {
        // Get indices of singular values to keep
        int* h_indices = new int[k];
        int idx = 0;
        for (int i = 0; i < min(m, n); ++i) {
            if (i < singular_values.size() && singular_values[i] > delta) {
                h_indices[idx++] = i;
            }
        }
        
        // Copy indices to device
        int* d_indices;
        cudaMalloc(&d_indices, k * sizeof(int));
        cudaMemcpy(d_indices, h_indices, k * sizeof(int), cudaMemcpyHostToDevice);
        
        // Extract truncated U
        for (int i = 0; i < k; ++i) {
            int col = h_indices[i];
            copyColumn<<<(m + 255) / 256, 256>>>(d_U, d_U_trunc, m, m, col, i, m, m);
            cudaDeviceSynchronize();
        }
        
        // Extract truncated V
        for (int i = 0; i < k; ++i) {
            int row = h_indices[i];
            copyRow<<<(n + 255) / 256, 256>>>(d_V, d_V_trunc, n, k, row, i, n, k);
            cudaDeviceSynchronize();
        }

        // Set truncated S
        cudaMemset(d_S_trunc, 0, k * k * sizeof(double));
        for (int i = 0; i < k; ++i) {
            if (i < singular_values.size()) {
                setDiagonalElement<<<1, 1>>>(d_S_trunc, singular_values[i], k, i);
                cudaDeviceSynchronize();
            }
        }
        
        delete[] h_indices;
        cudaFree(d_indices);
    }
    
    delete[] h_result;
    delete[] h_U;
    delete[] h_V;
    
    return k;
}

// Perform mode-n unfolding of tensor
void unfoldTensor(double* h_tensor, double* h_unfolded, int N, int n, const vector<int>& dims) {
    int total_size = 1;
    for (int i = 0; i < dims.size(); i++) {
        total_size *= dims[i];
    }
    
    // For n-mode unfolding, dim n becomes rows and the rest are flattened into columns
    int rows = dims[n];
    int cols = total_size / rows;
    
    // Calculate stride pattern
    vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i+1] * dims[i+1];
    }
    
    // Perform unfolding
    for (int i = 0; i < total_size; i++) {
        // Convert linear index to multi-index
        int remaining = i;
        vector<int> multi_index(dims.size());
        for (int j = 0; j < dims.size(); j++) {
            multi_index[j] = remaining / strides[j];
            remaining %= strides[j];
        }
        
        // For mode-n, the n-th dimension becomes the row
        int row = multi_index[n];
        
        // Calculate column index from the flattened other dimensions
        int col = 0;
        int col_stride = 1;
        for (int j = dims.size() - 1; j >= 0; j--) {
            if (j != n) {
                col += multi_index[j] * col_stride;
                col_stride *= dims[j];
            }
        }
        
        h_unfolded[row * cols + col] = h_tensor[i];
    }
}

// Perform TT-SVD decomposition
vector<double*> TT_SVD(double* h_tensor, int* dims, int ndims, double epsilon) {
    // Calculate total size
    int total_size = 1;
    for (int i = 0; i < ndims; i++) {
        total_size *= dims[i];
    }
    
    // Create vector for dimension sizes
    vector<int> dim_sizes(ndims);
    for (int i = 0; i < ndims; i++) {
        dim_sizes[i] = dims[i];
    }
    
    // Initialize result vector for cores
    vector<double*> cores;
    vector<int> ranks(ndims+1);
    ranks[0] = 1;  // r_0 = 1 by definition
    ranks[ndims] = 1; // r_d = 1 by definition
    
    // Calculate Frobenius norm of input tensor
    double* d_tensor;
    cudaMalloc(&d_tensor, total_size * sizeof(double));
    cudaMemcpy(d_tensor, h_tensor, total_size * sizeof(double), cudaMemcpyHostToDevice);
    
    double norm = frobeniusNormCUDA(d_tensor, total_size);
    double delta = (epsilon / sqrt(ndims - 1)) * norm;
    
    cout << "Tensor norm: " << norm << ", delta: " << delta << endl;
    
    // Working arrays
    double* current_tensor = new double[total_size];
    memcpy(current_tensor, h_tensor, total_size * sizeof(double));
    
    // Process each dimension
    for (int k = 0; k < ndims - 1; k++) {
        cout << "Processing dimension " << k << endl;
        
        // Calculate unfolding size
        int rows = dim_sizes[k];
        int cols = total_size / rows;
        
        // Mode-k unfolding
        double* unfolded = new double[rows * cols];
        unfoldTensor(current_tensor, unfolded, dim_sizes[k], k, dim_sizes);
        
        // Allocate device memory
        double *d_A, *d_U, *d_S, *d_V;
        cudaMalloc(&d_A, rows * cols * sizeof(double));
        cudaMalloc(&d_U, rows * rows * sizeof(double));
        cudaMalloc(&d_S, rows * cols * sizeof(double));
        cudaMalloc(&d_V, cols * cols * sizeof(double));
        
        // Allocate memory for truncated matrices
        double *d_U_trunc, *d_S_trunc, *d_V_trunc;
        cudaMalloc(&d_U_trunc, rows * rows * sizeof(double));
        cudaMalloc(&d_S_trunc, rows * cols * sizeof(double));
        cudaMalloc(&d_V_trunc, cols * cols * sizeof(double));
        
        // Copy unfolded matrix to device
        cudaMemcpy(d_A, unfolded, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
        
        // Perform truncated SVD
        int rank = deltaTruncatedSVDCUDA(d_A, d_U, d_S, d_V, d_U_trunc, d_S_trunc, d_V_trunc, rows, cols, delta);
        ranks[k+1] = rank;
        
        // Copy core data back to host
        double* core = new double[ranks[k] * rows * rank];
        double* h_U_trunc = new double[rows * rank];
        cudaMemcpy(h_U_trunc, d_U_trunc, rows * rank * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Reshape U_k to be the core G_k (r_{k-1} x n_k x r_k)
        if (k == 0) {
            // First core is special: 1 x n_1 x r_1
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rank; j++) {
                    core[i * rank + j] = h_U_trunc[i * rank + j];
                }
            }
        } else {
            // Intermediate cores are r_{k-1} x n_k x r_k
            // Need to reshape correctly based on previous rank
            // This is a simplified reshaping for demonstration
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rank; j++) {
                    for (int r = 0; r < ranks[k]; r++) {
                        core[r * rows * rank + i * rank + j] = h_U_trunc[i * rank + j] / ranks[k];
                    }
                }
            }
        }
        
        cores.push_back(core);
        cout << "Added core of shape [" << ranks[k] << ", " << rows << ", " << rank << "]" << endl;
        
        // Prepare matrix for next iteration: S * V^T
        if (k < ndims - 2) {
            double* h_S_trunc = new double[rank * rank];
            double* h_V_trunc = new double[cols * rank];
            
            cudaMemcpy(h_S_trunc, d_S_trunc, rank * rank * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_V_trunc, d_V_trunc, cols * rank * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Calculate S * V^T
            double* h_SV = new double[rank * cols];
            for (int i = 0; i < rank; i++) {
                for (int j = 0; j < cols; j++) {
                    h_SV[i * cols + j] = 0.0;
                    for (int r = 0; r < rank; r++) {
                        h_SV[i * cols + j] += h_S_trunc[i * rank + r] * h_V_trunc[j * rank + r];
                    }
                }
            }
            
            // Reshape SV to prepare for next iteration
            delete[] current_tensor;
            current_tensor = new double[total_size / rows * rank];
            
            // This reshape operation depends on the tensor structure
            // For simplicity, we're just copying the data linearly
            for (int i = 0; i < rank * cols; i++) {
                current_tensor[i] = h_SV[i];
            }
            
            // Update dimensions for next iteration
            total_size = total_size / rows * rank;
            dim_sizes[k] = rank;
            
            delete[] h_S_trunc;
            delete[] h_V_trunc;
            delete[] h_SV;
        }
        
        // Final core treatment for the last dimension
        if (k == ndims - 2) {
            // Last dimension handling: S*V^T becomes the last core
            double* h_S_trunc = new double[rank * rank];
            double* h_V_trunc = new double[cols * rank];
            
            cudaMemcpy(h_S_trunc, d_S_trunc, rank * rank * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_V_trunc, d_V_trunc, cols * rank * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Calculate S * V^T for the last core
            double* last_core = new double[rank * cols];
            for (int i = 0; i < rank; i++) {
                for (int j = 0; j < cols; j++) {
                    last_core[i * cols + j] = 0.0;
                    for (int r = 0; r < rank; r++) {
                        last_core[i * cols + j] += h_S_trunc[i * rank + r] * h_V_trunc[j * rank + r];
                    }
                }
            }
            
            cores.push_back(last_core);
            cout << "Added final core of shape [" << rank << ", " << dim_sizes[ndims-1] << ", " << 1 << "]" << endl;
            
            delete[] h_S_trunc;
            delete[] h_V_trunc;
        }
        
        // Clean up this iteration
        delete[] unfolded;
        delete[] h_U_trunc;
        
        cudaFree(d_A);
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_V);
        cudaFree(d_U_trunc);
        cudaFree(d_S_trunc);
        cudaFree(d_V_trunc);
    }
    
    delete[] current_tensor;
    cudaFree(d_tensor);
    
    return cores;
}

int main() {
    auto start = high_resolution_clock::now();
    
    // Set tensor dimensions for 5D tensor
    const int N = 13;  // Size of each dimension
    const int ndims = 5;  // Number of dimensions
    int dims[5] = {N, N, N, N, N};
    const int total_size = N * N * N * N * N;  // 243 elements for N=3
    
    // Create and initialize tensor on host
    double* h_tensor = new double[total_size];
    mt19937 gen(42);  // Fixed seed for reproducibility
    normal_distribution<double> dist(10.0, 1.0);
    
    for (int i = 0; i < total_size; i++) {
        h_tensor[i] = dist(gen);
    }
    
    // Perform TT-SVD
    double epsilon = 0.001;  // Truncation parameter
    vector<double*> cores = TT_SVD(h_tensor, dims, ndims, epsilon);
    
    cout << "TT-SVD decomposition complete." << endl;
    cout << "Number of cores: " << cores.size() << endl;
    
    // Display core information
    for (int i = 0; i < cores.size(); i++) {
        int r_prev = (i == 0) ? 1 : 0;  // Calculate from actual ranks
        int r_next = (i == cores.size() - 1) ? 1 : 0;  // Calculate from actual ranks
        
        cout << "Core " << i+1 << " shape: [" << r_prev << ", " << N << ", " << r_next << "]" << endl;
    }
    
    // Clean up
    delete[] h_tensor;
    for (double* core : cores) {
        delete[] core;
    }
    
    auto end = high_resolution_clock::now();
    double time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Time taken: " << time_taken << " milliseconds" << endl;
    
    return 0;
}