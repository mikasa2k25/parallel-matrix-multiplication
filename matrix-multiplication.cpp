#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

using Matrix = vector<vector<int>>;

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
    int n = A.size(), m = B[0].size(), p = B.size();
    Matrix C(n, vector<int>(m, 0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Worker function for each thread
void multiplyChunk(const Matrix& A, const Matrix& B, Matrix& C, int startRow, int endRow) {
    int cols = B[0].size(), sharedDim = B.size();
    for (int i = startRow; i < endRow; ++i)
        for (int j = 0; j < cols; ++j)
            for (int k = 0; k < sharedDim; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

// Multi-threaded version
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

    for (auto& t : threads)
        t.join();

    return C;
}

int main(int argc, char* argv[]) {
    // Default values
    int SIZE = 1000;
    int NUM_THREADS = 4;

    // Allow command-line override: ./matrix <input size> <input thread>
    if (argc >= 2) SIZE = atoi(argv[1]);
    if (argc >= 3) NUM_THREADS = atoi(argv[2]);

    cout << "Matrix size: " << SIZE << " x " << SIZE << "\n";
    cout << "Thread count: " << NUM_THREADS << "\n";

    // Generate random matrices A and B
    Matrix A = generateMatrix(SIZE, SIZE);
    Matrix B = generateMatrix(SIZE, SIZE);

    // Single-threaded multiplication
    cout << "\nRunning single-threaded multiplication...\n";
    auto startSingle = high_resolution_clock::now();
    Matrix C1 = multiplySingle(A, B);
    auto endSingle = high_resolution_clock::now();
    auto timeSingle = duration_cast<milliseconds>(endSingle - startSingle).count();
    cout << "Single-threaded time: " << timeSingle << " ms\n";

    // Multi-threaded multiplication
    cout << "\nRunning multi-threaded multiplication...\n";
    auto startMulti = high_resolution_clock::now();
    Matrix C2 = multiplyMulti(A, B, NUM_THREADS);
    auto endMulti = high_resolution_clock::now();
    auto timeMulti = duration_cast<milliseconds>(endMulti - startMulti).count();
    cout << "Multi-threaded time: " << timeMulti << " ms\n";

    // Speedup
    double speedup = double(timeSingle) / timeMulti;
    cout << "\nSpeedup (Single / Multi): " << speedup << "x\n";

    return 0;
}
