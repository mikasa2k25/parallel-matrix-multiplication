#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

using namespace std;

// Serial matrix multiplication
void serialMultiply(const vector<int>& A, const vector<int>& B, vector<int>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            int a = A[i * size + k];
            for (int j = 0; j < size; ++j) {
                C[i * size + j] += a * B[k * size + j];
            }
        }
    }
}

// Parallel block-wise multiplication
void parallelMultiply(const vector<int>& A, const vector<int>& B, vector<int>& C, 
                      int size, int startRow, int endRow, int blockSize = 64) {
    for (int i = startRow; i < endRow; i += blockSize) {
        for (int k = 0; k < size; k += blockSize) {
            for (int j = 0; j < size; j += blockSize) {
                for (int ii = i; ii < min(i + blockSize, endRow); ++ii) {
                    for (int kk = k; kk < min(k + blockSize, size); ++kk) {
                        int a = A[ii * size + kk];
                        for (int jj = j; jj < min(j + blockSize, size); ++jj) {
                            C[ii * size + jj] += a * B[kk * size + jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int size = 1024; // Large matrix for GitHub’s multi-core servers
    vector<int> A(size * size, 1);
    vector<int> B(size * size, 1);
    vector<int> C_serial(size * size, 0);
    vector<int> C_parallel(size * size, 0);

    // Serial execution
    auto start = chrono::high_resolution_clock::now();
    serialMultiply(A, B, C_serial, size);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> serialTime = end - start;

    // Parallel execution
    int numThreads = thread::hardware_concurrency(); // Use all available cores
    vector<thread> threads;
    int rowsPerThread = size / numThreads;
    start = chrono::high_resolution_clock::now();
    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? size : startRow + rowsPerThread;
        threads.emplace_back(parallelMultiply, cref(A), cref(B), ref(C_parallel), 
                            size, startRow, endRow);
    }
    for (auto& th : threads) th.join();
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parallelTime = end - start;

    // Results
    cout << "Matrix Size: " << size << "x" << size << endl;
    cout << "[Serial] Time: " << serialTime.count() << "s\n";
    cout << "[Parallel] Time: " << parallelTime.count() << "s\n";
    cout << "Speedup: " << serialTime.count() / parallelTime.count() << "x\n";
    return 0;
}
