#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>

using namespace std;

enum MatrixType {
    ZEROS,
    ONES,
    EYE,
    RAND
};

class Matrix {
    private:
        int rows;
        int cols;
        double *data;
        void checkBounds(int row, int col);
        void checkDimensions(Matrix other);

    public:
        Matrix(int rows, int cols, MatrixType type = ZEROS);
        double getValue(int row, int col);
        void setValue(int row, int col, double value);
        string getShape();
        void printValues();
        void initZeros();
        void initOnes();
        void initEye();
        void initRand();
        // Operations
        Matrix operator+(Matrix other);
        Matrix operator+(double scalar);
};

#endif
