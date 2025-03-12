#include <iostream>
#include <cmath>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;
using namespace xt;


double vectorNorm(const xarray<double>& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v(i) * v(i);
    }
    return sqrt(sum);
}


void householderBidiagonalization(xarray<double>& A, xarray<double>& U, xarray<double>& V) {
    int m = A.shape(0);
    int n = A.shape(1);
    
    U = eye<double>(m);
    V = eye<double>(n);

    for (int k = 0; k < min(m, n); ++k) {
        // Householder transformation for column
        xarray<double> x = view(A, range(k, m), k);
        double norm_x = vectorNorm(x);
        double alpha = (x(0) >= 0) ? -norm_x : norm_x;
        
        xarray<double> u = x;
        u(0) -= alpha;
        double norm_u = vectorNorm(u);
        if (norm_u > 1e-5) {
            u /= norm_u;
        
            // Apply transformation to A (I - 2uu^T) * A
            for (int j = k; j < n; ++j) {
                double dot_product = 0.0;
                for (size_t i = k; i < m; ++i) {
                    dot_product += u(i - k) * A(i, j);
                }
                for (size_t i = k; i < m; ++i) {
                    A(i, j) -= 2 * u(i - k) * dot_product;
                }
            }
        }

        if (k < n - 1) {
            // Householder transformation for row
            x = view(A, k, range(k + 1, n));
            norm_x = vectorNorm(x);
            alpha = (x(0) >= 0) ? -norm_x : norm_x;

            u = x;
            u(0) -= alpha;
            norm_u = vectorNorm(u);
            if (norm_u > 1e-5) {
                u /= norm_u;
                
                // Apply transformation to A: A * (I - 2uu^T)
                for (int i = 0; i < m; ++i) {
                    double dot_product = 0.0;
                    for (size_t j = k + 1; j < n; ++j) {
                        dot_product += A(i, j) * u(j - (k + 1));
                    }
                    for (size_t j = k + 1; j < n; ++j) {
                        A(i, j) -= 2 * dot_product * u(j - (k + 1));
                    }
                }
            }
        }
    }
}

// Extract Singular Values (QR Method Approximation)
xarray<double> extractSingularValues(xarray<double>& B) {
    int m = B.shape(0);
    int n = B.shape(1);
    xarray<double> Sigma = zeros<double>({m, n});
    
    for (int i = 0; i < min(m, n); ++i) {
        Sigma(i, i) = abs(B(i, i));  // Approximate singular values
    }

    return Sigma;
}

// Compute SVD (A = U * Σ * V^T)
void computeSVD(xarray<double>& A, xarray<double>& U, xarray<double>& Sigma, xarray<double>& V) {
    int m = A.shape(0);
    int n = A.shape(1);

    // Step 1: Bidiagonalization
    householderBidiagonalization(A, U, V);

    // Step 2: Extract Singular Values (approximate)
    Sigma = extractSingularValues(A);
}

xarray<double> unfold(xarray<double>& tensor,int mode,int td)
{
    vector<size_t> shape;
    shape.push_back(tensor.shape(mode-1));
    int d2 = 1;
    for(int i=0;i<td;i++)
    {
        if(i != mode-1) d2 *= tensor.shape(i);
    }
    shape.push_back(d2);

    vector<size_t> perm(tensor.shape().size());
    iota(perm.begin(),perm.end(),0);
    swap(perm[0],perm[mode]);

    xarray<double> tensorT = transpose(tensor,perm);

    xarray<double> unfolded_matrix = reshape_view(tensorT,shape);

    return unfolded_matrix;
}
// Function to truncate SVD components to keep only the 'k' largest singular values
tuple<xarray<double>, xarray<double>, xarray<double>> truncateSVD(
    const xarray<double>& U,
    const xarray<double>& sigma,
    const xarray<double>& V,
    int k) {
    // Get dimensions
    auto u_shape = U.shape();
    auto v_shape = V.shape();
    auto sigma_size = sigma.size();
    
    // Make sure k is not larger than the number of singular values
    k = min(k, static_cast<int>(sigma_size));
    
    // Create a copy of sigma and get indices
    xarray<double> sigma_copy = sigma;
    vector<size_t> indices(sigma_size);
    iota(indices.begin(), indices.end(), 0);
    
    // Sort indices based on singular values (in descending order)
    sort(indices.begin(), indices.end(),
         [&sigma_copy](size_t i1, size_t i2) {
             return sigma_copy(i1) > sigma_copy(i2);
         });
    
    // Take only the first k indices
    indices.resize(k);
    
    // Create truncated sigma
    xarray<double> truncated_sigma = zeros<double>({static_cast<size_t>(k)});
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        truncated_sigma(i) = sigma(indices[i]);
    }
    
    // Create truncated U (keep columns corresponding to top singular values)
    xarray<double> truncated_U = zeros<double>({u_shape[0], static_cast<size_t>(k)});
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < u_shape[0]; i++) {
        for (int j = 0; j < k; j++) {
            truncated_U(i, j) = U(i, indices[j]);
        }
    }
    
    // Create truncated V (keep rows corresponding to top singular values)
    xarray<double> truncated_V = zeros<double>({static_cast<size_t>(k), v_shape[1]});
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (size_t j = 0; j < v_shape[1]; j++) {
            truncated_V(i, j) = V(indices[i], j);
        }
    }
    
    return {truncated_U, truncated_sigma, truncated_V};
}


tuple<xarray<double>, xarray<double>, xarray<double>, int> deltatruncatedSVD(xarray<double>& C, double delta) {
    // First compute regular SVD
    xarray<double> U, S, V;
    computeSVD(C, U, S, V);
    
    // Extract diagonal values from S
    int min_dim = min(S.shape()[0], S.shape()[1]);
    vector<double> s_values(min_dim);

    #pragma omp parallel for
    for(int i = 0; i < min_dim; i++) {
        s_values[i] = S(i, i);
    }
    
    // Compute total squared sum
    double total_squared = 0;
    #pragma omp parallel for reduction(+:total_squared)
    for(int i = 0; i < min_dim; i++) {
        total_squared += s_values[i] * s_values[i];
    }
    
    // Find rank that satisfies the error bound
    double error_squared = 0;
    int rank = min_dim;
    
    for(int i = min_dim - 1; i >= 0; i--) {
        error_squared += s_values[i] * s_values[i];
        if(sqrt(error_squared) > delta) {
            rank = i + 1;
            break;
        }
    }
    
    // Convert singular values to vector form for truncation
    xarray<double> s_vec = zeros<double>({static_cast<size_t>(min_dim)});

    #pragma omp parallel for
    for(int i = 0; i < min_dim; i++) {
        s_vec(i) = s_values[i];
    }
    
    // Truncate to rank
    auto [U_trunc, S_trunc, V_trunc] = truncateSVD(U, s_vec, V, rank);
    
    return {U_trunc, S_trunc, V_trunc, rank};
}


double frobenius_norm(xarray<double>& A, int a, int b, int c, int d, int e)
{
    double res = 0.0;

    #pragma omp parallel for reduction(+:res) collapse(5)
    for(int i = 0; i < a; i++)
    {
        for(int j = 0; j < b; j++)
        {
            for(int k = 0; k < c; k++)
            {
                for(int l = 0; l < d; l++)
                {
                    for(int m = 0; m < e; m++)
                    {
                        res += A(i, j, k, l, m) * A(i, j, k, l, m);
                    }
                }
            }
        }
    }

    return sqrt(res);
}


vector<xarray<double>> TT_SVD(xarray<double>& A, double epsilon) {
    // Get dimensions
    auto shape = A.shape();
    int d = shape.size();
    cout<<d<<endl;
    // Initialize cores vector
    vector<xarray<double>> G(d);
    
    // Step 1: Initialization
    double normA = frobenius_norm(A,5,5,5,5,5);
    double delta = (epsilon / sqrt(d-1)) * normA;
    
    // Step 2: Initialize temporary tensor and r0
    xarray<double> C = A;
    int r_prev = 1;
    
    // Step 3: Main loop
    for(int k = 0; k < d-1; k++) {
        // Step 4: Reshape C
        int nk = shape[k];
        int numel = C.size();
        vector<size_t> reshape_dims = {static_cast<size_t>(r_prev * nk), 
                                     static_cast<size_t>(numel/(r_prev * nk))};
        C = reshape_view(C, reshape_dims);
        
        // Step 5: Compute delta-truncated SVD
        xarray<double> U, S, V;
        computeSVD(C, U, S, V);

        cout<<U.shape()[0]<<" "<<U.shape()[1]<<endl;
        cout<<S.shape()[0]<<" "<<S.shape()[1]<<endl;
        cout<<V.shape()[0]<<" "<<V.shape()[1]<<endl;
        
        // Find rank based on delta criterion
        double total_norm = 0;
        for(size_t i = 0; i < S.size(); i++) {
            total_norm += S(i) * S(i);
        }
        
        double running_error = 0;
        int rk = S.size();
        for(int i = S.size()-1; i >= 0; i--) {
            running_error += S(i) * S(i);
            if(sqrt(running_error) > delta) {
                rk = i + 1;
                break;
            }
        }
        if(rk == 0 && S.size() > 0) rk = 1;
        
        cout<<"Rank "<<rk<<endl;
        // Truncate SVD to rank rk
        tuple<xarray<double>, xarray<double>, xarray<double>> tup = truncateSVD(U, S, V, 1);
        xarray<double> U_trunc = get<0>(tup);
        xarray<double> S_trunc = get<1>(tup);
        xarray<double> V_trunc = get<2>(tup);
        
        // cout<<"HI"<<endl;
        // Step 5: Create new core Gk
        vector<size_t> core_shape = {static_cast<size_t>(r_prev), 
                                   static_cast<size_t>(nk), 
                                   static_cast<size_t>(rk)};
        G[k] = reshape_view(U_trunc, core_shape);
        
        
        // Step 5: Update C for next iteration
        xarray<double> SV = zeros<double>({static_cast<size_t>(rk), V_trunc.shape()[1]});
        for(int i = 0; i < rk; i++) {
            for(size_t j = 0; j < V_trunc.shape()[1]; j++) {
                SV(i,j) = S_trunc(i) * V_trunc(i,j);
            }
        }
        C = SV;
        r_prev = rk;
    }
    
    // Step 9: Set last core
    G[d-1] = C;
    vector<size_t> last_core_shape = {static_cast<size_t>(r_prev), 
                                     static_cast<size_t>(shape[d-1]), 
                                     static_cast<size_t>(1)};
    G[d-1] = reshape_view(G[d-1], last_core_shape);
    
    return G;
}

int main() {
    // Example matrix
    // xarray<double> A = {{3, 2, 2}, {2, 3, -2}};

    // xarray<double> U, Sigma, V;
    // computeSVD(A, U, Sigma, V);

    // // Output Results
    // cout << "U:\n" << U << "\n";
    // cout << "Sigma:\n" << Sigma << "\n";
    // cout << "V:\n" << V << "\n";

    xarray<double> A = random::randn<double>({8,8,8,8,8});
    // vector<size_t> newshape = {A.shape(0),A.shape(1)*A.shape(2)*A.shape(3)*A.shape(4)};
    // xarray<double> unfolded_along1 = reshape_view(A, newshape);
    xarray<double> unfolded = unfold(A,2,5);
    // cout<<unfolded.shape(0)<<" "<<unfolded.shape(1);
    // xarray<double> U, Sigma, V;
    // computeSVD(unfolded,U,Sigma,V);
    // cout<<U.shape()[0]<<" "<<U.shape()[1]<<endl;
    // cout<<Sigma.shape()[0]<<" "<<Sigma.shape()[1]<<endl;
    // cout<<V.shape()[0]<<" "<<Sigma.shape()[1]<<endl;
    // tuple<xarray<double>, xarray<double>, xarray<double>> T = truncateSVD(U, Sigma, V,3);
    // xarray<double> truncated_U = get<0>(T);
    // xarray<double> truncated_sigma = get<1>(T);
    // xarray<double> truncated_V = get<2>(T);
    // cout<<"------"<<endl;
    // cout<<truncated_U.shape()[0]<<" "<<truncated_U.shape()[1]<<endl;
    // cout<<truncated_sigma.shape()[0]<<" "<<truncated_sigma.shape()[1]<<endl;
    // cout<<truncated_V.shape()[0]<<" "<<truncated_V.shape()[1]<<endl;
    
    // double fnorm = frobenius_norm(A,5,5,5,5,5);
    // cout<<fnorm<<endl;



    // cout<<U.shape()[0]<<" "<<U.shape()[1]<<endl;
    // cout<<Sigma.shape()[0]<<" "<<Sigma.shape()[1]<<endl;
    // cout<<V.shape()[0]<<" "<<Sigma.shape()[1]<<endl;
    tuple<xarray<double>, xarray<double>, xarray<double>, int> tup = deltatruncatedSVD(unfolded,0.0001);
    xarray<double> truncated_U = get<0>(tup);
    xarray<double> truncated_sigma = get<1>(tup);
    xarray<double> truncated_V = get<2>(tup);

    cout<<truncated_U.shape()[0]<<" "<<truncated_U.shape()[1]<<endl;
    cout<<truncated_sigma.shape()[0]<<" "<<truncated_sigma.shape()[1]<<endl;
    cout<<truncated_V.shape()[0]<<" "<<truncated_V.shape()[1]<<endl;

    vector<xarray<double>> res = TT_SVD(A,0.001);

    cout<<res[0].size()<<endl;
    return 0;
}
