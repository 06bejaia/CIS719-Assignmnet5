#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

const int INF = 1e9;

void floyd_openmp(vector<vector<int>> &dist) {
    int n = dist.size();
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

int main() {
    int n = 4;
    vector<vector<int>> dist = {
        {0, 3, INF, 7},
        {8, 0, 2, INF},
        {5, INF, 0, 1},
        {2, INF, INF, 0}
    };

    floyd_openmp(dist);

    cout << "Distance Matrix:" << endl;
    for (auto &row : dist) {
        for (int val : row) {
            if (val == INF) cout << "INF ";
            else cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
