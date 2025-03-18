#include <iostream>
#include <cmath>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda.h>

// #define N 10
#define EPSILON 1e-9 

using namespace std;

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

    int blockSize = 1024;
    int gridSize = (total_size + blockSize - 1)/ blockSize;

    vectorNorm<<<gridSize, blockSize>>>(d_tensor, d_result, total_size);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

void matrixMultiplyCUDA(double* d_A, double* d_B, double* d_C, int m, int k, int n) {

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x,
        (m + blockDim.y - 1) / blockDim.y
    );

    // printf("Grid dimension: %d, Block dimension: %d",256, ((n + blockDim.x - 1) / blockDim.x) * ((m + blockDim.y - 1) / blockDim.y));
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
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

    int blockSize = 256;

    for (int k = 0; k < min(m, n); ++k) {
        extractColumn<<<(m-k + 255) / 256, 256>>>(d_A, d_x, m, n, k, k, ldA);

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

            int numBlocksU = (m + blockSize - 1) / blockSize;
            updateUMatrix<<<numBlocksU, blockSize>>>(d_U, d_u, m, k, ldU);

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

void reshapeTensorCUDA(double* d_input, double* d_output, int mode, int N, int total_size) {

    cudaMemcpy(d_output, d_input, total_size * sizeof(double), cudaMemcpyDeviceToDevice);
}

void compute_singular_values(double *d, double *e, int n, vector<double> &singular_values) {
    for (int i = 0; i < n; i++) {
        singular_values[i] = fabs(d[i]); // Initial guess: diagonal values
    }

    while (true) {
        // Check for convergence (all off-diagonal elements small enough)
        bool converged = true;
        for (int i = 0; i < n - 1; i++) {
            if (fabs(e[i]) > EPSILON) {
                converged = false;
                break;
            }
        }
        if (converged) break;

        // Wilkinson shift
        double mu = singular_values[n - 1] * singular_values[n - 1];
        for (int i = 0; i < n - 1; i++) {
            double t = (singular_values[i] * singular_values[i]) - mu;
            double s = sqrt(t * t + e[i] * e[i]);
            double c = t / s;
            double s_rot = e[i] / s;

            // Apply Givens rotation
            double temp = singular_values[i] * c + e[i] * s_rot;
            e[i] = -singular_values[i] * s_rot + e[i] * c;
            singular_values[i] = temp;
        }
    }
}

// Extract bidiagonal components from h_result
void extract_bidiagonal(double *h_result, double *d, double *e, int m, int n) {
    for (int i = 0; i < m; i++) {
        d[i] = h_result[i * n + i]; // Main diagonal
        if (i < m - 1) {
            e[i] = h_result[i * n + i + 1]; // Superdiagonal
        }
    }
}

// Function to return singular values
vector<double> get_singular_values(double *h_result, int m, int n) {
    double *d = new double[m];
    double *e = new double[m - 1];
    vector<double> singular_values(m);

    extract_bidiagonal(h_result, d, e, m, n);
    compute_singular_values(d, e, m, singular_values);

    delete[] d;
    delete[] e;
    return singular_values;
}


int deltaTruncatedSVDCUDA(double* d_A, double* d_U, double* d_S, double* d_V, double* d_U_trunc, double* d_S_trunc, double* d_V_trunc, int m, int n, double delta, int rank) {

    cudaError_t error;

    householderBidiagonalizationCUDA(d_A, d_U, d_V, m, n, n, m, n);
    printf("Bidiagonalization done\n\n\n");

    double* h_result = new double[m * n];
    double* h_U = new double[m * m];
    double* h_V = new double[n * n];
    
    cudaMemcpy(h_result, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    vector<double> singular_values = get_singular_values(h_result, m, n);

    for(double s_value: singular_values) cout<<s_value<<"|||";

    printf("\n\nSingular values extracted\n\n");
    
    printf("HII\n\n");
    
    // Step 2: Apply Delta Truncation
    // Copy singular values to host
    
    int k = 0;
  
    for(double sv: singular_values)
    {
      if(sv > delta) k++;
    }
    printf("K value: %d\n\n", k);

    if (k > 0) {

        int* h_indices = new int[k];
        
        int idx = 0;
        for (int i = 0; i < min(m, n); ++i) {
            if (singular_values[i] > delta) {
                h_indices[idx++] = i;
            }
        }
        
        int* d_indices;
        cudaMalloc(&d_indices, k * sizeof(int));
        cudaMemcpy(d_indices, h_indices, k * sizeof(int), cudaMemcpyHostToDevice);
        
        for (int i = 0; i < k; ++i) {
            int col = h_indices[i];
            copyColumn<<<(m + 255) / 256, 256>>>(d_U, d_U_trunc, m, m, col, i, m, m);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
            }
            
        }
        
        for (int i = 0; i < k; ++i) {
            int row = h_indices[i];
            copyRow<<<(n + 255) / 256, 256>>>(d_V, d_V_trunc, n, n, row, i, n, k);

            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
            }
        }

        cudaMemset(d_S_trunc, 0, k * k * sizeof(double));
        for (int i = 0; i < k; ++i) {
            int idx = h_indices[i];
            setDiagonalElement<<<1, 1>>>(d_S_trunc, singular_values[idx], k, i);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
            }
        }
        

        delete[] h_indices;
        cudaFree(d_indices);
    }

    printf("Doneee");
    return k;
}

vector<double*> TT_SVD(double* h_tensor, int size, double epsilon,int N)
{
    double* d_tensor;
    cudaMalloc(&d_tensor, size * sizeof(double));
    cudaMemcpy(d_tensor, h_tensor, size * sizeof(double), cudaMemcpyHostToDevice);

    double norm = frobeniusNormCUDA(d_tensor, size);

    double delta = (epsilon/2) *  norm;

    cout<<"Delta = "<<delta<<endl;

    vector<int> r(6,8);
    r[0] = 1;
    vector<double*> cores;
    int m = N;
    int n = pow(N,4);

    double *d_A, *d_U, *d_S, *d_V;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_U, m * m * sizeof(double));
    cudaMalloc(&d_S, m * n * sizeof(double));
    cudaMalloc(&d_V, n * n * sizeof(double));
    

    double *d_U_trunc, *d_S_trunc, *d_V_trunc;
    cudaMalloc(&d_U_trunc, m * m * sizeof(double));
    cudaMalloc(&d_S_trunc, m * n * sizeof(double));
    cudaMalloc(&d_V_trunc, n * n * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_A, h_tensor, m * n * sizeof(double), cudaMemcpyHostToDevice);

    int rank = -1;

    for(int k=1; k<=4; k++)
    {
        rank = deltaTruncatedSVDCUDA(d_A, d_U, d_S, d_V, d_U_trunc, d_S_trunc, d_V_trunc, m, n, delta, rank);
        
        // Correct memory allocation (remove sizeof(double) multiplication)
        double *G = new double[m*m];
        
        // Add error checking for cudaMemcpy
        cudaError_t err = cudaMemcpy(G, d_U_trunc, m*m*sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cout << "CUDA memory copy failed: " << cudaGetErrorString(err) << endl;
            delete[] G;  // Clean up in case of error
            continue;  // Skip this iteration
        }
        
        cores.push_back(G);
        cout << "Core " << k << " added. Cores vector size: " << cores.size() << endl;
    }
    
    return cores;
}

int main() {
    // Small 5D tensor (2x2x2x2x2)
    const int N = 2;
    const int total_size = N * N * N * N * N; // 32 elements
    
    // Create and initialize tensor on host
    double* h_tensor = new double[total_size];
    mt19937 gen(42); // Fixed seed for reproducibility
    normal_distribution<double> dist(10.0, 1.0);
    
    for (int i = 0; i < total_size; i++) {
        h_tensor[i] = dist(gen);
    }
    
    // Mode-1 unfolding: (2, 16)
    int m = N;
    int n = N * N * N * N;
    
    // Print original matrix (first few elements)
    cout << "Original matrix (first few elements):" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < min(5, n); j++) {
            cout << fixed << setprecision(4) << h_tensor[i * n + j] << " ";
        }
        cout << endl;
    }
    
    vector<double*> cores = TT_SVD(h_tensor,m*n,0.001,N);
    cout<<"Number of cores = "<<cores.size();

    // // Make a copy of the original for verification
    // double* h_A_copy = new double[m * n];
    // memcpy(h_A_copy, h_tensor, m * n * sizeof(double));

    // double *d_A, *d_U, *d_S, *d_V;
    // cudaMalloc(&d_A, m * n * sizeof(double));
    // cudaMalloc(&d_U, m * m * sizeof(double));
    // cudaMalloc(&d_S, m * n * sizeof(double));
    // cudaMalloc(&d_V, n * n * sizeof(double));
    

    // double *d_U_trunc, *d_S_trunc, *d_V_trunc;
    // cudaMalloc(&d_U_trunc, m * m * sizeof(double));
    // cudaMalloc(&d_S_trunc, m * n * sizeof(double));
    // cudaMalloc(&d_V_trunc, n * n * sizeof(double));
    
    // // Copy data to device
    // cudaMemcpy(d_A, h_tensor, m * n * sizeof(double), cudaMemcpyHostToDevice);

    // double delta = 0.5;

    // int rank = -1;


    // std::cout << "Running delta-truncated SVD..." << std::endl;
    // rank = deltaTruncatedSVDCUDA(d_A, d_U, d_S, d_V, d_U_trunc, d_S_trunc, d_V_trunc, m, n, delta, rank);

    // double* h_S_trunc = new double[m*n*sizeof(double)];
    // cudaMemcpy(h_S_trunc, d_S_trunc, m*n * sizeof(double), cudaMemcpyDeviceToHost);

    // std::cout << "Truncated singular values: ";
    // for (int i = 0; i < rank; i++) {
    //     std::cout << h_S_trunc[i*m + i] << " ";
    // }
    // cout<<endl;

    // double* h_U_trunc;
    // double* h_V_trunc;

    // h_U_trunc = new double[m*m*sizeof(double)];
    // h_V_trunc = new double[n*n*sizeof(double)];

    // cudaMemcpy(h_U_trunc,d_U_trunc, m*m*sizeof(double),cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_V_trunc,d_V_trunc, n*n*sizeof(double), cudaMemcpyDeviceToHost);

    // for(int i=0;i<m*m;i++)
    // {
    //    cout<<h_U_trunc[i]<<" ";
    // }

    // cout<<endl;
    
    // for(int i=0;i<n*n;i++)
    // {
    //   cout<<h_V_trunc[i]<<" ";
    // }
    
    // Run bidiagonalization
    // householderBidiagonalizationCUDA(d_A, d_U, d_V, m, n, n, m, n);
    
    // Copy results back
    // double* h_result = new double[m * n];
    // double* h_U = new double[m * m];
    // double* h_V = new double[n * n];
    
    // cudaMemcpy(h_result, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_U, d_U, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_V, d_V, n * n * sizeof(double), cudaMemcpyDeviceToHost);
     
    // // Print bidiagonal matrix (should only have nonzeros on diagonal and superdiagonal)
    // cout << "\nBidiagonal matrix result:" << endl;

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < min(5, n); j++) {
    //         cout << fixed << setprecision(4) << h_result[i * n + j] << " ";
    //     }
    //     cout << endl;
    // }

    // vector<double> singular_values = get_singular_values(h_result, m, n);

    // for(double s_value: singular_values) cout<<s_value<<"|||";

    // // Verify U is orthogonal by checking U*U^T
    // cout << "\nVerifying U is orthogonal (U*U^T):" << endl;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < m; j++) {
    //         double sum = 0.0;
    //         for (int k = 0; k < m; k++) {
    //             sum += h_U[i * m + k] * h_U[j * m + k];
    //         }
    //         cout << fixed << setprecision(4) << sum << " ";
    //     }
    //     cout << endl;
    // }
    
    // // Print part of U matrix
    // cout << "\nU matrix:" << endl;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < m; j++) {
    //         cout << fixed << setprecision(4) << h_U[i * m + j] << " ";
    //     }
    //     cout << endl;
    // }
    
    // // Print part of V matrix (just first few columns)
    // cout << "\nV matrix (first few columns):" << endl;
    // for (int i = 0; i < min(3, n); i++) {
    //     for (int j = 0; j < min(3, n); j++) {
    //         cout << fixed << setprecision(4) << h_V[i * n + j] << " ";
    //     }
    //     cout << endl;
    // }
    
    // // Clean up
    // delete[] h_tensor;
    // delete[] h_A_copy;
    // delete[] h_result;
    // delete[] h_U;
    // delete[] h_V;
    // cudaFree(d_A);
    // cudaFree(d_U);
    // cudaFree(d_V);
    
    return 0;
}
