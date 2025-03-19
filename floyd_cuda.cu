#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

const int INF = 1e9;

// Kernel function to perform Floyd-Warshall on GPU
__global__ void floyd_cuda_kernel(int *dist, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int via_k = dist[i * n + k] + dist[k * n + j];
        dist[i * n + j] = min(dist[i * n + j], via_k);  // Use the built-in min function
    }
}

void floyd_cuda(vector<vector<int>> &dist) {
    int n = dist.size();
    int *d_dist;
    int size = n * n * sizeof(int);

    // Convert the 2D vector to a 1D array for CUDA processing
    int *h_dist = new int[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_dist[i * n + j] = dist[i][j];
        }
    }

    // Allocate memory on the device
    cudaMalloc(&d_dist, size);
    cudaMemcpy(d_dist, h_dist, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    for (int k = 0; k < n; ++k) {
        floyd_cuda_kernel<<<numBlocks, threadsPerBlock>>>(d_dist, k, n);
        cudaDeviceSynchronize();
    }

    // Copy the result back to host
    cudaMemcpy(h_dist, d_dist, size, cudaMemcpyDeviceToHost);

    // Convert the 1D array back to 2D vector
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist[i][j] = h_dist[i * n + j];
        }
    }

    // Free the memory on the device
    cudaFree(d_dist);
    delete[] h_dist;
}

int main() {
    int n = 4;
    vector<vector<int>> dist = {
        {0, 3, INF, 7},
        {8, 0, 2, INF},
        {5, INF, 0, 1},
        {2, INF, INF, 0}
    };

    floyd_cuda(dist);

    cout << "Shortest path matrix: " << endl;
    for (auto &row : dist) {
        for (int val : row) {
            if (val == INF) cout << "INF ";
            else cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
