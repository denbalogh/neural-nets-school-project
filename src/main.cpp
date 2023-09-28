#include <iostream>
#include <chrono>
#include "matrix/matrix.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if(argc == 2 && string(argv[1]) == "aura"){
        cout << "Aura has 128 cores" << endl;
        // Aura has 128 cores
        Matrix::setMaxThreads(128);
    }

    cout << "Matmul efficiency test:" << endl;
    for(int i = 10; i <= 1000; i += 50) {
        Matrix A = Matrix(i, i, RAND);
        cout << "Matrix shape: " << A.getShape() << endl;

        auto start = high_resolution_clock::now();
        Matrix B = A.matmul(A);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Elapsed time REGULAR: " << duration.count() << "ms" << endl;

        start = high_resolution_clock::now();
        Matrix C = A.matmul_parallel(A);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "Elapsed time PARALLEL: " << duration.count() << "ms" << endl;

        if(B.compareValues(C)){
            cout << "They are the SAME" << endl;
        } else {
            cout << "They are NOT the SAME" << endl;
        }
    }

    // Matrix A = Matrix(100, 100, RAND);

    // Matrix matmul_parallel = A.matmul_parallel(A);
    // Matrix matmul = A.matmul(A);

    // if(matmul_parallel.compareValues(matmul)){
    //     cout << "They are the SAME" << endl;
    // } else {
    //     cout << "They are NOT the SAME" << endl;
    // }

    // cout << endl << "Parallel:" << endl;
    // matmul_parallel.printValues();
    // cout << endl << "Regular:" << endl;
    // matmul.printValues();

    return 0;
}
