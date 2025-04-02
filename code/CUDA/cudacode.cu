#include <iostream>
#include <cmath>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda.h>

#define EPSILON 1e-9 

using namespace std;
using namespace chrono;

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

    return sqrt(result);
}

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

void householderBidiagonalizationCUDA(double* d_A, double* d_U, double* d_V, int m, int n, int ldA, int ldU, int ldV) {
    double* d_u;
    double* d_x;
    cudaMalloc(&d_u, max(m, n) * sizeof(double));
    cudaMalloc(&d_x, max(m, n) * sizeof(double));

    double* h_u = new double[max(m, n)];
    double* h_x = new double[max(m, n)];

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

    for (int k = 0; k < min(m, n); ++k) {
        extractColumn<<<(m-k + 255) / 256, 256>>>(d_A, d_x, m, n, k, k, ldA);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_x, d_x, (m-k) * sizeof(double), cudaMemcpyDeviceToHost);

        double norm_x = 0.0;
        for (int i = 0; i < m-k; ++i) {
            norm_x += h_x[i] * h_x[i];
        }
        norm_x = sqrt(norm_x);

        if (norm_x < 1e-10) continue;

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
            cudaDeviceSynchronize();

            int numBlocksU = (m + blockSize - 1) / blockSize;
            updateUMatrix<<<numBlocksU, blockSize>>>(d_U, d_u, m, k, ldU);
            cudaDeviceSynchronize();
        }

        if (k < n - 1) {
            extractRow<<<(n-k-1 + 255) / 256, 256>>>(d_A, d_x, m, n, k, k+1, ldA);
            cudaDeviceSynchronize();
            
            cudaMemcpy(h_x, d_x, (n-k-1) * sizeof(double), cudaMemcpyDeviceToHost);

            norm_x = 0.0;
            for (int i = 0; i < n-k-1; ++i) {
                norm_x += h_x[i] * h_x[i];
            }
            norm_x = sqrt(norm_x);

            if (norm_x < 1e-10) continue;

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
                cudaDeviceSynchronize();

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

void reshapeTensorCUDA(double* d_input, double* d_output, int rows, int cols, int new_rows, int new_cols) {
    // For TT-SVD, we need to properly reshape the data for the next core computation
    cudaMemcpy(d_output, d_input, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
}

void extract_bidiagonal(double *h_result, double *d, double *e, int m, int n) {
    for (int i = 0; i < min(m, n); i++) {
        d[i] = h_result[i * n + i];
        if (i < min(m, n) - 1) {
            e[i] = h_result[i * n + i + 1];
        } else {
            e[i] = 0.0;
        }
    }
}

void compute_singular_values(double *d, double *e, int n, vector<double> &singular_values) {

    for (int i = 0; i < n; i++) {
        singular_values[i] = fabs(d[i]);
    }


    for (int iter = 0; iter < 30; iter++) {
        bool converged = true;
        for (int i = 0; i < n - 1; i++) {
            if (fabs(e[i]) > EPSILON * (fabs(d[i]) + fabs(d[i+1]))) {
                converged = false;
                
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
    
    sort(singular_values.begin(), singular_values.end(), greater<double>());
}

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

int deltaTruncatedSVDCUDA(double* d_A, double* d_U, double* d_S, double* d_V, double* d_U_trunc, double* d_S_trunc, double* d_V_trunc, int m, int n, double delta) {

    householderBidiagonalizationCUDA(d_A, d_U, d_V, m, n, n, m, n);
    
    double* h_result = new double[m * n];
    double* h_U = new double[m * m];
    double* h_V = new double[n * n];
    
    cudaMemcpy(h_result, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    vector<double> singular_values = get_singular_values(h_result, m, n);
    
    cout << "Computed singular values: ";
    for (int i = 0; i < min(5, (int)singular_values.size()); i++) {
        cout << singular_values[i] << " ";
    }
    cout << "..." << endl;
    
    int k = 0;
    for (double sv : singular_values) {
        if (sv > delta) k++;
    }
    
    cout << "Keeping " << k << " singular values above threshold " << delta << endl;

    if (k > 0) {
        int* h_indices = new int[k];
        int idx = 0;
        for (int i = 0; i < min(m, n); ++i) {
            if (i < singular_values.size() && singular_values[i] > delta) {
                h_indices[idx++] = i;
            }
        }
        
        int* d_indices;
        cudaMalloc(&d_indices, k * sizeof(int));
        cudaMemcpy(d_indices, h_indices, k * sizeof(int), cudaMemcpyHostToDevice);
        
        for (int i = 0; i < k; ++i) {
            int col = h_indices[i];
            copyColumn<<<(m + 255) / 256, 256>>>(d_U, d_U_trunc, m, m, col, i, m, m);
            cudaDeviceSynchronize();
        }
        
        for (int i = 0; i < k; ++i) {
            int row = h_indices[i];
            copyRow<<<(n + 255) / 256, 256>>>(d_V, d_V_trunc, n, k, row, i, n, k);
            cudaDeviceSynchronize();
        }

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

void unfoldTensor(double* h_tensor, double* h_unfolded, int N, int n, const vector<int>& dims) {
    int total_size = 1;
    for (int i = 0; i < dims.size(); i++) {
        total_size *= dims[i];
    }
    
    int rows = dims[n];
    int cols = total_size / rows;
    
    vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i+1] * dims[i+1];
    }
    
    for (int i = 0; i < total_size; i++) {
        int remaining = i;
        vector<int> multi_index(dims.size());
        for (int j = 0; j < dims.size(); j++) {
            multi_index[j] = remaining / strides[j];
            remaining %= strides[j];
        }
        
        int row = multi_index[n];
        
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

vector<double*> TT_SVD(double* h_tensor, int* dims, int ndims, double epsilon) {

    int total_size = 1;
    for (int i = 0; i < ndims; i++) {
        total_size *= dims[i];
    }
    

    vector<int> dim_sizes(ndims);
    for (int i = 0; i < ndims; i++) {
        dim_sizes[i] = dims[i];
    }
    

    vector<double*> cores;
    vector<int> ranks(ndims+1);
    ranks[0] = 1;
    ranks[ndims] = 1;
    

    double* d_tensor;
    cudaMalloc(&d_tensor, total_size * sizeof(double));
    cudaMemcpy(d_tensor, h_tensor, total_size * sizeof(double), cudaMemcpyHostToDevice);
    
    double norm = frobeniusNormCUDA(d_tensor, total_size);
    double delta = (epsilon / sqrt(ndims - 1)) * norm;
    
    cout << "Tensor norm: " << norm << ", delta: " << delta << endl;
    
    double* current_tensor = new double[total_size];
    memcpy(current_tensor, h_tensor, total_size * sizeof(double));
    
    for (int k = 0; k < ndims - 1; k++) {
        cout << "Processing dimension " << k << endl;
        
        int rows = dim_sizes[k];
        int cols = total_size / rows;
        
        double* unfolded = new double[rows * cols];
        unfoldTensor(current_tensor, unfolded, dim_sizes[k], k, dim_sizes);
        
        double *d_A, *d_U, *d_S, *d_V;
        cudaMalloc(&d_A, rows * cols * sizeof(double));
        cudaMalloc(&d_U, rows * rows * sizeof(double));
        cudaMalloc(&d_S, rows * cols * sizeof(double));
        cudaMalloc(&d_V, cols * cols * sizeof(double));
        
        double *d_U_trunc, *d_S_trunc, *d_V_trunc;
        cudaMalloc(&d_U_trunc, rows * rows * sizeof(double));
        cudaMalloc(&d_S_trunc, rows * cols * sizeof(double));
        cudaMalloc(&d_V_trunc, cols * cols * sizeof(double));
        
        cudaMemcpy(d_A, unfolded, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
        
        int rank = deltaTruncatedSVDCUDA(d_A, d_U, d_S, d_V, d_U_trunc, d_S_trunc, d_V_trunc, rows, cols, delta);
        ranks[k+1] = rank;
        
        double* core = new double[ranks[k] * rows * rank];
        double* h_U_trunc = new double[rows * rank];
        cudaMemcpy(h_U_trunc, d_U_trunc, rows * rank * sizeof(double), cudaMemcpyDeviceToHost);
        
        if (k == 0) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rank; j++) {
                    core[i * rank + j] = h_U_trunc[i * rank + j];
                }
            }
        } else {
            // Intermediate cores are r_{k-1} x n_k x r_k
            // Need to reshape correctly based on previous rank
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
    
    const int N = 13; 
    const int ndims = 5; 
    int dims[5] = {N, N, N, N, N};
    const int total_size = N * N * N * N * N; 

    double* h_tensor = new double[total_size];
    mt19937 gen(42); 
    normal_distribution<double> dist(10.0, 1.0);
    
    for (int i = 0; i < total_size; i++) {
        h_tensor[i] = dist(gen);
    }
    

    double epsilon = 0.001;
    vector<double*> cores = TT_SVD(h_tensor, dims, ndims, epsilon);
    
    cout << "TT-SVD decomposition complete." << endl;
    cout << "Number of cores: " << cores.size() << endl;
    
    for (int i = 0; i < cores.size(); i++) {
        int r_prev = (i == 0) ? 1 : 0;
        int r_next = (i == cores.size() - 1) ? 1 : 0;
        
        cout << "Core " << i+1 << " shape: [" << r_prev << ", " << N << ", " << r_next << "]" << endl;
    }
    
    delete[] h_tensor;
    for (double* core : cores) {
        delete[] core;
    }
    
    auto end = high_resolution_clock::now();
    double time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Time taken: " << time_taken << " milliseconds" << endl;
    
    return 0;
}