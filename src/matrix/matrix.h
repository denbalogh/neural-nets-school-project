#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cmath>

// #define DEBUG 1

using namespace std;

enum MatrixType {
    ZEROS,
    ONES,
    RAND
};

class Matrix {
    private:
        int rows;
        int cols;
        double *data;

        #ifdef DEBUG
            void checkBounds(int row, int col, string message) const;
            void checkDimensions(const Matrix& other, string operation) const;
        #endif

        static int MAX_THREADS;

    public:
        Matrix();
        Matrix(int rows, int cols, MatrixType type = ZEROS);
        static void setMaxThreads(int max_threads);
        int getRows() const;
        int getCols() const;
        double get(int row, int col) const;
        void set(int row, int col, double value);

        #ifdef DEBUG
            string getShape() const;
            void printValues() const;
        #endif

        bool isEqualTo(const Matrix& other) const;
        void initZeros();
        void initOnes();
        void initRand();
        // Operations
        Matrix matmul(const Matrix& other) const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator+(double value) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator-(double value) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(double value) const;
        Matrix operator/(const Matrix& other) const;
        Matrix operator/(double value) const;
        Matrix pow(double power) const;
        Matrix exp() const;
        Matrix log() const;
        Matrix sum(int dim) const;
        Matrix max(int dim) const;
        Matrix argmax(int dim) const;
        Matrix mean(int dim) const;
        Matrix transpose() const;
        Matrix tanh() const;
        Matrix softmax() const;
};

void matmulThread(const Matrix& A, const Matrix& B, int row_start, int col_start, int ops_num, Matrix& result);

#endif
