#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cmath>
#include <random>

#include "../debug.h"

using namespace std;

enum MatrixType {
    ZEROS,
    ONES,
};

class Matrix {
    private:
        int rows = 0;
        int cols = 0;
        vector<float> data;

        #ifdef DEBUG
            void checkBounds(int row, int col, string message) const;
            void checkDimensions(const Matrix& other, string operation) const;
        #endif

        static int MAX_THREADS;

    public:
        Matrix(){};
        Matrix(int rows, int cols, MatrixType type = ZEROS);
        Matrix(int rows, int cols, float mean, float std);
        static void setMaxThreads(int max_threads);
        int getRows() const { return rows; };
        int getCols() const { return cols; };
        float get(int row, int col) const;
        void set(int row, int col, float value);
        Matrix clone() const;

        #ifdef DEBUG
            string getShape() const;
            void printValues() const;
            bool isEqualTo(const Matrix& other) const;
        #endif

        void initNormal(float mean, float std);
        // Operations
        Matrix matmul(const Matrix& other) const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator+(float value) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator-(float value) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(float value) const;
        Matrix operator/(const Matrix& other) const;
        Matrix operator/(float value) const;
        Matrix pow(float power) const;
        Matrix exp() const;
        Matrix log() const;
        Matrix sum(int dim) const;
        Matrix max(int dim) const;
        Matrix argmax(int dim) const;
        Matrix mean(int dim) const;
        Matrix std(int dim) const;
        Matrix transpose() const;
        Matrix normalize() const;
        Matrix relu() const;
        Matrix dRelu() const;
        Matrix softmax() const;
};

void matmulThread(const Matrix& A, const Matrix& B, int row_start, int col_start, int ops_num, Matrix& result);

#endif
