#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cmath>

using namespace std;

enum MatrixType {
    ZEROS,
    ONES,
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
        Matrix();
        Matrix(int rows, int cols, MatrixType type = ZEROS);
        static void setMaxThreads(int max_threads);
        int getRows();
        int getCols();
        double getValue(int row, int col);
        void setValue(int row, int col, double value);
        string getShape();
        void printValues();
        bool compareValues(Matrix& other);
        void initZeros();
        void initOnes();
        void initRand();
        // Operations
        Matrix matmul(Matrix& other);
        Matrix operator+(Matrix& other);
        Matrix relu();
        Matrix softmax();
};

void matmulThread(Matrix& A, Matrix& B, int row_start, int col_start, int ops_num, Matrix& result);

#endif
