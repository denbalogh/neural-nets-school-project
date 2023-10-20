#include "matrix.h"

// static variables and functions
int Matrix::MAX_THREADS;

void Matrix::setMaxThreads(int max_threads){
    Matrix::MAX_THREADS = max_threads;
}

#ifdef DEBUG
    // helper functions
    void Matrix::checkBounds(int r, int c, string message) const{
        if(r < 0 || r >= rows || c < 0 || c >= cols) {
            throw invalid_argument("Matrix index out of bounds, " + message);
        }
    }

    void Matrix::checkDimensions(const Matrix& other, string operation) const{
        if(rows != other.getRows() || cols != other.getCols()) {
            throw invalid_argument("Matrix dimensions must match. " + getShape() + " != " + other.getShape() + ", " + operation);
        }
    }

    string Matrix::getShape() const{
        return "(" + to_string(rows) + ", " + to_string(cols) + ")";
    }

    void Matrix::printValues() const{
        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < cols; c++) {
                cout << get(r, c) << " ";
            }
            cout << endl;
        }
    }
#endif

int Matrix::getRows() const{
    return rows;
}

int Matrix::getCols() const{
    return cols;
}

Matrix::Matrix() : rows(0), cols(0), data(NULL) {}

Matrix::Matrix(int r, int c, MatrixType type) {

    #ifdef DEBUG
        if(r < 1 || c < 1) {
            throw invalid_argument("Matrix dimensions must be positive and non-zero");
        }
    #endif

    rows = r;
    cols = c;
    data = new double[r * c]{ 0.0 }; // Initialize all values to 0.0

    switch(type) {
        case ONES:
            initOnes();
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
            set(r, c, 0.0);
        }
    }
}

void Matrix::initOnes() {
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            set(r, c, 1.0);
        }
    }
}

void Matrix::initRand() {
    srand(time(NULL));

    // initialize random values between -0.01 and 0.01
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            set(r, c, (double)rand() / RAND_MAX * 0.02 - 0.01);
        }
    }
}

double Matrix::get(int r, int c) const{
    #ifdef DEBUG
        checkBounds(r, c, "getValue: " + to_string(r) + ", " + to_string(c));
    #endif

    return data[r * cols + c];
}

void Matrix::set(int r, int c, double value) {
    #ifdef DEBUG
        checkBounds(r, c, "setValue: " + to_string(r) + ", " + to_string(c));
    #endif

    data[r * cols + c] = value;
}

bool Matrix::isEqualTo(const Matrix& other) const{
    #ifdef DEBUG
        if(rows != other.getRows() || cols != other.getCols()) {
            return false;
        }
    #endif

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            if(get(r, c) != other.get(r, c)) {
                return false;
            }
        }
    }

    return true;
}

void matmulThread(const Matrix& A, const Matrix& B, int row_start, int col_start, int ops_num, Matrix& result){
    for(int i = 0; i < ops_num; i++){
        int row = row_start + (col_start + i) / B.getCols();
        int col = (col_start + i) % B.getCols();
        double sum = 0.0;

        for(int j = 0; j < A.getCols(); j++){
            sum += A.get(row, j) * B.get(j, col);
        }

        result.set(row, col, sum);
    }
}

Matrix Matrix::matmul(const Matrix& other) const{
    #ifdef DEBUG
        if(cols != other.getRows()) {
            throw invalid_argument("Matmul A @ B: cols of A must match rows of B");
        }
    #endif

    Matrix result(rows, other.getCols());
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

Matrix Matrix::operator+(const Matrix& other) const{
    // Adding a row vector, need to broadcast
    if(other.getRows() == 1){
        #ifdef DEBUG
            if(other.getCols() != cols){
                throw invalid_argument("When adding a vector to matrix, cols must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) + other.get(0, c));
            }
        }

        return result;
    }

    // Adding a column vector, need to broadcast
    if(other.getCols() == 1){
        #ifdef DEBUG
            if(other.getRows() != rows){
                throw invalid_argument("When adding a column vector to matrix, rows must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) + other.get(r, 0));
            }
        }

        return result;
    }

    // Adding a matrix

    #ifdef DEBUG
        checkDimensions(other, "A + B");
    #endif

    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) + other.get(r, c));
        }
    }

    return result;
}

Matrix Matrix::operator+(double value) const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) + value);
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const{
    // Subtracting a row vector, need to broadcast
    if(other.getRows() == 1){
        #ifdef DEBUG
            if(other.getCols() != cols){
                throw invalid_argument("When subtracting a vector from matrix, cols must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) - other.get(0, c));
            }
        }

        return result;
    }

    // Subtracting a column vector, need to broadcast
    if(other.getCols() == 1){
        #ifdef DEBUG
            if(other.getRows() != rows){
                throw invalid_argument("When subtracting a column vector from matrix, rows must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) - other.get(r, 0));
            }
        }

        return result;
    }

    // Subtracting a matrix

    #ifdef DEBUG
        checkDimensions(other, "A - B");
    #endif

    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) - other.get(r, c));
        }
    }

    return result;
}

Matrix Matrix::operator-(double value) const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) - value);
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const{
    // Multiplying a row vector, need to broadcast
    if(other.getRows() == 1){
        #ifdef DEBUG
            if(other.getCols() != cols){
                throw invalid_argument("When multiplying a vector to matrix, cols must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) * other.get(0, c));
            }
        }

        return result;
    }

    // Multiplying a column vector, need to broadcast
    if(other.getCols() == 1){
        #ifdef DEBUG
            if(other.getRows() != rows){
                throw invalid_argument("When multiplying a column vector to matrix, rows must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) * other.get(r, 0));
            }
        }

        return result;
    }

    // Multiplying a matrix

    #ifdef DEBUG
        checkDimensions(other, "A * B");
    #endif

    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) * other.get(r, c));
        }
    }

    return result;
}

Matrix Matrix::operator*(double value) const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) * value);
        }
    }

    return result;
}

Matrix Matrix::operator/(const Matrix& other) const{
    // Dividing a row vector, need to broadcast
    if(other.getRows() == 1){
        #ifdef DEBUG
            if(other.getCols() != cols){
                throw invalid_argument("When dividing a vector from matrix, cols must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) / other.get(0, c));
            }
        }

        return result;
    }

    // Dividing a column vector, need to broadcast
    if(other.getCols() == 1){
        #ifdef DEBUG
            if(other.getRows() != rows){
                throw invalid_argument("When dividing a column vector from matrix, rows must match");
            }
        #endif

        Matrix result(rows, cols);

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                result.set(r, c, get(r, c) / other.get(r, 0));
            }
        }

        return result;
    }

    // Dividing a matrix

    #ifdef DEBUG
        checkDimensions(other, "A / B");
    #endif

    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) / other.get(r, c));
        }
    }

    return result;
}

Matrix Matrix::operator/(double value) const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) / value);
        }
    }

    return result;
}

Matrix Matrix::pow(double power) const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, std::pow(get(r, c), power));
        }
    }

    return result;
}

Matrix Matrix::exp() const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, std::exp(get(r, c)));
        }
    }

    return result;
}

Matrix Matrix::log() const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, std::log(get(r, c)));
        }
    }

    return result;
}

Matrix Matrix::sum(int dim) const{
    if(dim == 0){
        Matrix result(1, cols);

        for(int c = 0; c < cols; c++){
            double sum = 0.0;

            for(int r = 0; r < rows; r++){
                sum += get(r, c);
            }

            result.set(0, c, sum);
        }

        return result;
    }else if(dim == 1){
        Matrix result(rows, 1);

        for(int r = 0; r < rows; r++){
            double sum = 0.0;

            for(int c = 0; c < cols; c++){
                sum += get(r, c);
            }

            result.set(r, 0, sum);
        }

        return result;
    }else{
        throw invalid_argument("Sum dim must be 0 or 1");
    }
}

Matrix Matrix::max(int dim) const{
    if(dim == 0){
        Matrix result(1, cols);

        for(int c = 0; c < cols; c++){
            double max = get(0, c);

            for(int r = 1; r < rows; r++){
                if(get(r, c) > max){
                    max = get(r, c);
                }
            }

            result.set(0, c, max);
        }

        return result;
    }else if(dim == 1){
        Matrix result(rows, 1);

        for(int r = 0; r < rows; r++){
            double max = get(r, 0);

            for(int c = 1; c < cols; c++){
                if(get(r, c) > max){
                    max = get(r, c);
                }
            }

            result.set(r, 0, max);
        }

        return result;
    }else{
        throw invalid_argument("Max dim must be 0 or 1");
    }
}

Matrix Matrix::argmax(int dim) const{
    if(dim == 0){
        Matrix result(1, cols);

        for(int c = 0; c < cols; c++){
            double max = get(0, c);
            int argmax = 0;

            for(int r = 1; r < rows; r++){
                if(get(r, c) > max){
                    max = get(r, c);
                    argmax = r;
                }
            }

            result.set(0, c, argmax);
        }

        return result;
    }else if(dim == 1){
        Matrix result(rows, 1);

        for(int r = 0; r < rows; r++){
            double max = get(r, 0);
            int argmax = 0;

            for(int c = 1; c < cols; c++){
                if(get(r, c) > max){
                    max = get(r, c);
                    argmax = c;
                }
            }

            result.set(r, 0, argmax);
        }

        return result;
    }else{
        throw invalid_argument("Argmax dim must be 0 or 1");
    }
}

Matrix Matrix::mean(int dim) const{
    if(dim == 0){
        return sum(0) / rows;
    }else if(dim == 1){
        return sum(1) / cols;
    }else{
        throw invalid_argument("Mean dim must be 0 or 1");
    }
}

Matrix Matrix::transpose() const{
    Matrix result(cols, rows);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(c, r, get(r, c));
        }
    }

    return result;
}

Matrix Matrix::tanh() const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, std::tanh(get(r, c)));
        }
    }

    return result;
}

Matrix Matrix::softmax() const{
    Matrix max_row_values = max(1);
    Matrix values_minus_max = *this - max_row_values;
    Matrix exp_values_minus_max = values_minus_max.exp();
    Matrix sum_exp_values_minus_max = exp_values_minus_max.sum(1);
    Matrix result = exp_values_minus_max / sum_exp_values_minus_max;

    return result;
}
