#include "matrix.h"

// My laptop has 6 cores
int Matrix::MAX_THREADS = 6;

void Matrix::setMaxThreads(int max_threads){
    Matrix::MAX_THREADS = max_threads;
}

// helper functions
void Matrix::checkBounds(int r, int c, string message) {
    if(r < 0 || r >= rows || c < 0 || c >= cols) {
        throw invalid_argument("Matrix index out of bounds, " + message);
    }
}

void Matrix::checkDimensions(Matrix& other) {
    if(rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match");
    }
}

// public functions

int Matrix::getRows() {
    return rows;
}

int Matrix::getCols() {
    return cols;
}

Matrix::Matrix() {
    rows = 0;
    cols = 0;
    data = NULL;
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
    checkBounds(r, c, "getValue: " + to_string(r) + ", " + to_string(c));
    return data[r * cols + c];
}

void Matrix::setValue(int r, int c, double value) {
    checkBounds(r, c, "setValue: " + to_string(r) + ", " + to_string(c));
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

bool Matrix::compareValues(Matrix& other) {
    if(rows != other.rows || cols != other.cols) {
        return false;
    }

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            if(getValue(r, c) != other.getValue(r, c)) {
                return false;
            }
        }
    }

    return true;
}

Matrix Matrix::matmul(Matrix& other) {
    if(cols != other.rows) {
        throw invalid_argument("Matmul A @ B: cols of A must match rows of B");
    }

    Matrix result(rows, other.cols);

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < other.cols; c++) {
            double sum = 0.0;

            for(int i = 0; i < cols; i++) {
                sum += getValue(r, i) * other.getValue(i, c);
            }

            result.setValue(r, c, sum);
        }
    }

    return result;
}

void matmulThread(Matrix& A, Matrix& B, int row_start, int col_start, int ops_num, Matrix& result){
    for(int i = 0; i < ops_num; i++){
        int row = row_start + (col_start + i) / B.getCols();
        int col = (col_start + i) % B.getCols();
        double sum = 0.0;

        for(int j = 0; j < A.getCols(); j++){
            sum += A.getValue(row, j) * B.getValue(j, col);
        }

        result.setValue(row, col, sum);
    }
}

Matrix Matrix::matmulParallel(Matrix& other){
    if(cols != other.rows) {
        throw invalid_argument("Matmul A @ B: cols of A must match rows of B");
    }

    Matrix result(rows, other.cols);
    vector<thread> threads;

    const int ops = getRows() * other.getCols();
    const int ops_per_thread = ops / MAX_THREADS;
    const int ops_per_thread_remainder = ops % MAX_THREADS;

    int ops_done = 0;

    for(int i = 0; i < MAX_THREADS; i++){
        int ops_actual = i == 0 ? ops_per_thread + ops_per_thread_remainder : ops_per_thread;
        int row_start = i == 0 ? 0 : ops_done / other.getCols();
        int col_start = i == 0 ? 0 : ops_done % other.getCols();

        threads.push_back(thread(matmulThread, ref(*this), ref(other), row_start, col_start, ops_actual, ref(result)));
        ops_done += ops_actual;
    }

    for(int i = 0; i < MAX_THREADS; i++){
        threads[i].join();
    }
    
    return result;
}
