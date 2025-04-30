#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int SIZE = 500; // You can increase this for better performance comparisons

using Matrix = vector<vector<int>>;

Matrix generateMatrix(int rows, int cols) {
    Matrix mat(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = rand() % 10;
    return mat;
}

Matrix singleThreadedMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size(), m = B[0].size(), p = B.size();
    Matrix C(n, vector<int>(m, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

void multiplyPartial(const Matrix& A, const Matrix& B, Matrix& C, int startRow, int endRow) {
    int m = B[0].size(), p = B.size();
    for (int i = startRow; i < endRow; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

Matrix multiThreadedMultiply(const Matrix& A, const Matrix& B, int numThreads) {
    int n = A.size();
    Matrix C(n, vector<int>(B[0].size(), 0));
    vector<thread> threads;
    int chunkSize = n / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == numThreads - 1) ? n : start + chunkSize;
        threads.emplace_back(multiplyPartial, ref(A), ref(B), ref(C), start, end);
    }

    for (auto& th : threads)
        th.join();

    return C;
}

int main() {
    Matrix A = generateMatrix(SIZE, SIZE);
    Matrix B = generateMatrix(SIZE, SIZE);

    cout << "Starting single-threaded multiplication...\n";
    auto start1 = high_resolution_clock::now();
    Matrix C1 = singleThreadedMultiply(A, B);
    auto end1 = high_resolution_clock::now();
    cout << "Single-threaded time: " 
         << duration_cast<milliseconds>(end1 - start1).count() << " ms\n";

    cout << "Starting multi-threaded multiplication...\n";
    auto start2 = high_resolution_clock::now();
    Matrix C2 = multiThreadedMultiply(A, B, thread::hardware_concurrency());
    auto end2 = high_resolution_clock::now();
    cout << "Multi-threaded time: " 
         << duration_cast<milliseconds>(end2 - start2).count() << " ms\n";

    return 0;
}
