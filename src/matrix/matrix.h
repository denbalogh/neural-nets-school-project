#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <thread>

using namespace std;

enum MatrixType {
    ZEROS,
    ONES,
    EYE,
    RAND
};

class Matrix {
    private:
        double *data;
        int rows;
        int cols;
        void checkBounds(int row, int col, string message);
        void checkDimensions(Matrix& other);
        static int MAX_THREADS;

    public:
        static void setMaxThreads(int max_threads);
        int getRows();
        int getCols();
        Matrix(int rows, int cols, MatrixType type = ZEROS);
        double getValue(int row, int col);
        void setValue(int row, int col, double value);
        string getShape();
        void printValues();
        bool compareValues(Matrix& other);
        void initZeros();
        void initOnes();
        void initEye();
        void initRand();
        // Operations
        Matrix matmul(Matrix& other);
        Matrix matmul_parallel(Matrix& other);
};

void matmul_thread(Matrix& A, Matrix& B, int row_start, int col_start, int ops_num, Matrix& result);

#endif
