#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>

const int INF = 1e9; // Use a large value to represent infinity

void floyd_mpi(std::vector<std::vector<int>>& dist, int n, int rank, int size) {
    int chunk_size = n / size;  // Divide matrix rows equally among processes
    int start_row = rank * chunk_size;
    int end_row = (rank + 1) * chunk_size;
    
    // Ensure that the last process gets the remaining rows if n is not perfectly divisible
    if (rank == size - 1) {
        end_row = n;
    }

    // Perform the Floyd-Warshall algorithm
    for (int k = 0; k < n; ++k) {
        // Each process updates its portion of the matrix
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < n; ++j) {
                dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }

        // Synchronize all processes before moving to the next iteration
        MPI_Barrier(MPI_COMM_WORLD);

        // Share updated rows with all other processes
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                // Send rows to the other processes and receive rows from them
                MPI_Sendrecv(&dist[start_row][0], chunk_size * n, MPI_INT, p, 0,
                             &dist[start_row][0], chunk_size * n, MPI_INT, p, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 4;  // Example matrix size
    std::vector<std::vector<int>> dist = {
        {0, 3, INF, 7},
        {8, 0, 2, INF},
        {5, INF, 0, 1},
        {2, INF, INF, 0}
    };

    floyd_mpi(dist, n, rank, size);

    // Gather and print the final matrix after all processes have completed their work
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << dist[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
