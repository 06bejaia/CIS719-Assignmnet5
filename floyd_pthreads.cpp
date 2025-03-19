#include <iostream>
#include <vector>
#include <pthread.h>

using namespace std;

const int INF = 1e9;
vector<vector<int>> *dist;

struct ThreadData {
    int k;
    int i;
};

void* floyd_pthread(void* arg) {
    struct ThreadData* data = (struct ThreadData*) arg;
    int k = data->k;
    int i = data->i;
    int n = dist->size();  // Use -> to access the size() method of a pointer

    for (int j = 0; j < n; ++j) {
        (*dist)[i][j] = min((*dist)[i][j], (*dist)[i][k] + (*dist)[k][j]);
    }
    return nullptr;
}

void floyd_pthreads(vector<vector<int>> &dist_) {
    dist = &dist_;  // Set dist to point to the input vector
    int n = dist->size();  // Use -> to access the size() method of a pointer
    pthread_t threads[n][n];
    struct ThreadData thread_data[n][n];

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            thread_data[i][k].k = k;
            thread_data[i][k].i = i;
            pthread_create(&threads[i][k], nullptr, floyd_pthread, (void*)&thread_data[i][k]);
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                pthread_join(threads[i][j], nullptr);
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

    floyd_pthreads(dist);

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
