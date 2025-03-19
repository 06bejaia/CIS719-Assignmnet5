#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
const int INF = 1e9;

// Renaming the custom min function to avoid conflict with standard library min
__host__ __device__ inline int custom_min(int a, int b) {
    return (a < b) ? a : b;
}

__global__ void floyd_thrust_kernel(int *dist, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int via_k = dist[i * n + k] + dist[k * n + j];
        dist[i * n + j] = custom_min(dist[i * n + j], via_k);
    }
}

void floyd_thrust(vector<vector<int>> &dist) {
    int n = dist.size();
    thrust::host_vector<int> h_dist(n * n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            h_dist[i * n + j] = dist[i][j];

    thrust::device_vector<int> d_dist = h_dist;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    for (int k = 0; k < n; ++k) {
        floyd_thrust_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_dist.data()), k, n);
        cudaDeviceSynchronize();
    }

    h_dist = d_dist;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dist[i][j] = h_dist[i * n + j];
}

int main() {
    int n = 4;
    vector<vector<int>> dist = {
        {0, 3, INF, 7},
        {8, 0, 2, INF},
        {5, INF, 0, 1},
        {2, INF, INF, 0}
    };

    floyd_thrust(dist);

    for (auto &row : dist) {
        for (int val : row) {
            if (val == INF) cout << "INF ";
            else cout << val << " ";
        }
        cout << endl;
    }
    return 0;
}

