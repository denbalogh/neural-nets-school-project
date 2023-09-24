#include "matrix.h"

// helper functions
void Matrix::checkBounds(int r, int c) {
    if(r < 0 || r >= rows || c < 0 || c >= cols) {
        throw invalid_argument("Matrix index out of bounds");
    }
}

void Matrix::checkDimensions(Matrix other) {
    if(rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match");
    }
}

Matrix::Matrix(int r, int c, MatrixType type) {
    if(r < 1 || c < 1) {
        throw invalid_argument("Matrix dimensions must be positive");
    }

    rows = r;
    cols = c;
    data = new double[r * c]{ 0.0 }; // Initialize all values to 0.0

    switch(type) {
        case ONES:
            initOnes();
            break;
        case EYE:
            initEye();
            break;
        case RAND:
            initRand();
            break;
        default:
            break;
    }
}

void Matrix::initZeros() {
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            setValue(r, c, 0.0);
        }
    }
}

void Matrix::initOnes() {
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            setValue(r, c, 1.0);
        }
    }
}

void Matrix::initEye() {
    if(rows != cols) {
        throw invalid_argument("Matrix must be square");
    }

    initZeros();

    for(int i = 0; i < rows; i++) {
        setValue(i, i, 1.0);
    }
}

void Matrix::initRand() {
    srand(time(NULL));

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            setValue(r, c, (double) rand() / RAND_MAX);
        }
    }
}

double Matrix::getValue(int r, int c) {
    checkBounds(r, c);
    return data[r * cols + c];
}

void Matrix::setValue(int r, int c, double value) {
    checkBounds(r, c);
    data[r * cols + c] = value;
}

string Matrix::getShape() {
    return "(" + to_string(rows) + ", " + to_string(cols) + ")";
}

void Matrix::printValues() {
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            cout << getValue(r, c) << " ";
        }
        cout << endl;
    }
}

Matrix Matrix::operator+(Matrix other) {
    checkDimensions(other);

    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            result.setValue(r, c, getValue(r, c) + other.getValue(r, c));
        }
    }

    return result;
}

Matrix Matrix::operator+(double scalar) {
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            result.setValue(r, c, getValue(r, c) + scalar);
        }
    }

    return result;
}
