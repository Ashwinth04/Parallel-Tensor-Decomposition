#include <iostream>
#include <cmath>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;
using namespace xt;
using namespace chrono;

#define N 10

const double S_min = 10.0, S_max = 200.0;   // Stock price range
const double sigma_min = 0.1, sigma_max = 1.0;  // Volatility range
const double r_min = 0.01, r_max = 0.1;   // Interest rate range
const double T = 1.0;  // Time to maturity (1 year)
const double K_min = 50.0, K_max = 150.0;  // Strike price range

const int S_steps = N, sigma_steps = N, r_steps = N, t_steps = N, K_steps = N; // Grid resolution
const double dt = T / (t_steps - 1);  // Time step

// Function to initialize Black-Scholes discretized tensor
xarray<double> discretizeBlackScholes() 
{
    xarray<double> tensor = zeros<double>({S_steps, sigma_steps, r_steps, t_steps, K_steps});

    // Define grid points
    xarray<double> S_grid = linspace(S_min, S_max, S_steps);
    xarray<double> sigma_grid = linspace(sigma_min, sigma_max, sigma_steps);
    xarray<double> r_grid = linspace(r_min, r_max, r_steps);
    xarray<double> t_grid = linspace(static_cast<double>(0), T, t_steps);
    xarray<double> K_grid = linspace(K_min, K_max, K_steps);

    // Discretization using explicit finite difference method
    for (int s = 1; s < S_steps - 1; s++) {
        for (int v = 1; v < sigma_steps - 1; v++) {
            for (int r = 1; r < r_steps - 1; r++) {
                for (int t = 1; t < t_steps - 1; t++) {
                    for (int k = 1; k < K_steps - 1; k++) {
                        double S = S_grid[s], sigma = sigma_grid[v], r_val = r_grid[r], K = K_grid[k];

                        double dS = (S_max - S_min) / (S_steps - 1);
                        double dVdt = (tensor(s, v, r, t + 1, k) - tensor(s, v, r, t - 1, k)) / (2 * dt);
                        double dVdS = (tensor(s + 1, v, r, t, k) - tensor(s - 1, v, r, t, k)) / (2 * dS);
                        double d2VdS2 = (tensor(s + 1, v, r, t, k) - 2 * tensor(s, v, r, t, k) + tensor(s - 1, v, r, t, k)) / (dS * dS);

                        tensor(s, v, r, t, k) = dVdt + 0.5 * sigma * sigma * S * S * d2VdS2 + r_val * S * dVdS - r_val * tensor(s, v, r, t, k);
                    }
                }
            }
        }
    }
    return tensor;
}



double vectorNorm(xarray<double>& v) {
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

            // Update U
            for (int i = 0; i < m; ++i) {
                double dot_product = 0.0;
                for (size_t j = k; j < m; ++j) {
                    dot_product += U(i, j) * u(j - k);
                }
                for (size_t j = k; j < m; ++j) {
                    U(i, j) -= 2 * dot_product * u(j - k);
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

                // Update V
                for (int i = 0; i < n; ++i) {
                    double dot_product = 0.0;
                    for (size_t j = k + 1; j < n; ++j) {
                        dot_product += V(i, j) * u(j - (k + 1));
                    }
                    for (size_t j = k + 1; j < n; ++j) {
                        V(i, j) -= 2 * dot_product * u(j - (k + 1));
                    }
                }
            }
        }
    }
}

// Extract Singular Values from Bidiagonal Matrix
xarray<double> extractSingularValues(xarray<double>& B) {
    int m = B.shape(0);
    int n = B.shape(1);
    xarray<double> Sigma = zeros<double>({m, n});

    for (int i = 0; i < min(m, n); ++i) {
        Sigma(i, i) = abs(B(i, i));  // Approximate singular values
    }

    return Sigma;
}

void deltaTruncatedSVD(xarray<double>& A, xarray<double>& U, xarray<double>& Sigma, xarray<double>& V, double delta) {
    int m = A.shape(0);
    int n = A.shape(1);

    // Step 1: Compute SVD
    householderBidiagonalization(A, U, V);  // Ensure this provides proper decomposition
    Sigma = extractSingularValues(A); // Extract singular values from bidiagonal form

    // Step 2: Apply Delta Truncation
    vector<size_t> indices;
    for (size_t i = 0; i < Sigma.shape(0); ++i) {
        if (Sigma(i, i) > delta) {
            indices.push_back(i);
        }
    }

    // Step 3: Keep only truncated U, Sigma, V
    U = xt::view(U, xt::all(), xt::keep(indices));  // Keep relevant columns of U
    V = xt::view(V, xt::keep(indices), xt::all());  // Keep relevant rows of V
    Sigma = xt::view(Sigma, xt::keep(indices), xt::keep(indices));  // Keep relevant singular values
}

xarray<double> matmul(xarray<double>& A, xarray<double>& B) {
    size_t m = A.shape()[0];
    size_t k = A.shape()[1];
    size_t n = B.shape()[1];

    xarray<double> C = zeros<double>({m, n});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t p = 0; p < k; ++p) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
    return C;
}

double frobenius_norm(xarray<double>& A, int a, int b, int c, int d, int e)
{
    double res = 0.0;


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


vector<xarray<double>> TT_SVD(xarray<double>& A, double epsilon)
{
    double delta = (epsilon/2)*frobenius_norm(A,N,N,N,N,N);
    xarray<double> C = A;
    vector<int> r(6,8);
    r[0] = 1;
    vector<xarray<double>> cores;

    for(int k=1;k<=4;k++)
    {
        vector<size_t> cshape;
        cshape.push_back(r[k-1]*N);
        cshape.push_back(pow(N,4)/r[k-1]);
        C = reshape_view(C,cshape);
        xarray<double> U,S,V;
        deltaTruncatedSVD(C,U,S,V,delta);

        vector<size_t> ushape;
        ushape.push_back(r[k-1]);
        ushape.push_back(N);
        ushape.push_back(r[k]);
        xarray<double> Gk = reshape_view(U,ushape);
        cores.push_back(Gk);
        xarray<double> K = transpose(V);
        C = matmul(S,K);
    }

    return cores;
}



xarray<double> unfold(xarray<double>& tensor, int mode, int td) {
    vector<size_t> shape;
    shape.push_back(tensor.shape(mode - 1));
    int d2 = 1;
    for (int i = 0; i < td; i++) {
        if (i != mode - 1) d2 *= tensor.shape(i);
    }
    shape.push_back(d2);

    vector<size_t> perm(tensor.shape().size());
    iota(perm.begin(), perm.end(), 0);
    swap(perm[0], perm[mode]);

    xarray<double> tensorT = transpose(tensor, perm);

    xarray<double> unfolded_matrix = reshape_view(tensorT, shape);

    return unfolded_matrix;
}

int main() {
    // Generate a random 5D tensor
    auto start = high_resolution_clock::now();
    xarray<double> A = random::randn<double>({N, N, N, N, N});

    // xarray<double> A = discretizeBlackScholes();

    // Unfold the tensor along a mode (example: mode-2 to mode-5)
    xarray<double> unfolded = unfold(A, 2, 5);

    cout << unfolded.shape(0) << " " << unfolded.shape(1) << endl;

    // Variables for truncated SVD
    xarray<double> U, V, Sigma;
    double delta = 1e-8;  // Threshold for truncation

    // Perform delta-truncated SVD
    cout << "Unfolding done. Computing delta-truncated SVD" << endl;
    deltaTruncatedSVD(unfolded, U, Sigma, V, delta);

    cout << "U shape: " << U.shape(0) << " x " << U.shape(1) << endl;
    cout << "Sigma shape: " << Sigma.shape(0) << " x " << Sigma.shape(1) << endl;
    cout << "V shape: " << V.shape(0) << " x " << V.shape(1) << endl;

    for(int i=0;i<U.shape(0);i++)
    {
        for(int j=0;j<U.shape(1);j++)
        {
            cout<<U(i,j)<<" ";
        }
        cout<<endl;
    }


    vector<xarray<double>> cores = TT_SVD(A,0.001);

    for(int i=0;i<cores.size();i++)
    {
        cout<<cores[i].shape(0)<<" "<<cores[i].shape(1)<<endl;
        for(int j=0;j<cores[i].shape(0);j++)
        {
            for(int k=0;k<cores[i].shape(1);k++)
            {
                for(int l=0;l<cores[i].shape(2);l++)
                {
                    cout<<cores[i](j,k,l)<<" ";
                }
            }
        }
        cout<<endl;
    }
    auto end = high_resolution_clock::now();
    double time_taken = duration_cast<milliseconds>(end - start).count();
    cout<<"\n\nTime taken = "<<time_taken<<" milliseconds"<<endl;
    return 0;
}
