#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

using Matrix = vector<vector<int>>;

const int SIZE = 1000;  // Matrix size (NxN)
const int NUM_THREADS = 4;  // You can adjust this

// Generate a random matrix of size rows x cols
Matrix generateMatrix(int rows, int cols) {
    Matrix mat(rows, vector<int>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 10);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = dist(gen);
    return mat;
}

// Single-threaded matrix multiplication
Matrix multiplySingle(const Matrix& A, const Matrix& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    Matrix C(n, vector<int>(m, 0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// Worker function for a chunk of rows
void multiplyChunk(const Matrix& A, const Matrix& B, Matrix& C, int rowStart, int rowEnd) {
    int m = B[0].size();
    int p = B.size();

    for (int i = rowStart; i < rowEnd; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

// Multi-threaded matrix multiplication
Matrix multiplyMulti(const Matrix& A, const Matrix& B, int numThreads) {
    int n = A.size();
    Matrix C(n, vector<int>(B[0].size(), 0));

    vector<thread> threads;
    int chunkSize = n / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == numThreads - 1) ? n : start + chunkSize;
        threads.emplace_back(multiplyChunk, cref(A), cref(B), ref(C), start, end);
    }

    for (auto& th : threads)
        th.join();

    return C;
}

int main() {
    cout << "Generating matrices of size " << SIZE << "x" << SIZE << "...\n";
    Matrix A = generateMatrix(SIZE, SIZE);
    Matrix B = generateMatrix(SIZE, SIZE);

    // Single-threaded multiplication
    cout << "Running single-threaded multiplication...\n";
    auto start1 = high_resolution_clock::now();
    Matrix C1 = multiplySingle(A, B);
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1).count();
    cout << "Single-threaded time: " << duration1 << " ms\n";

    // Multi-threaded multiplication
    cout << "Running multi-threaded multiplication with " << NUM_THREADS << " threads...\n";
    auto start2 = high_resolution_clock::now();
    Matrix C2 = multiplyMulti(A, B, NUM_THREADS);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2 - start2).count();
    cout << "Multi-threaded time: " << duration2 << " ms\n";

    return 0;
}
