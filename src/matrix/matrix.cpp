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
#endif

Matrix::Matrix(int r, int c, MatrixType type): rows(r), cols(c){
    #ifdef DEBUG
        if(r < 1 || c < 1) {
            throw invalid_argument("Matrix dimensions must be positive and non-zero");
        }
    #endif

    data = vector<float>(r * c, type == ONES ? 1.0 : 0.0);
}

Matrix::Matrix(int r, int c, float mean, float std): rows(r), cols(c){
    #ifdef DEBUG
        if(r < 1 || c < 1) {
            throw invalid_argument("Matrix dimensions must be positive and non-zero");
        }
    #endif

    data = vector<float>(r * c, 0.0);
    initNormal(mean, std);
}

void Matrix::initNormal(float mean, float std) {
    default_random_engine generator;
    normal_distribution<float> distribution(mean, std);

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            set(r, c, distribution(generator));
        }
    }
}

float Matrix::get(int r, int c) const{
    #ifdef DEBUG
        checkBounds(r, c, "getValue: " + to_string(r) + ", " + to_string(c));
    #endif

    return data[r * cols + c];
}

void Matrix::set(int r, int c, float value) {
    #ifdef DEBUG
        checkBounds(r, c, "setValue: " + to_string(r) + ", " + to_string(c));
    #endif

    data[r * cols + c] = value;
}

Matrix Matrix::clone() const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            result.set(r, c, get(r, c));
        }
    }

    return result;
}

void matmulThread(const Matrix& A, const Matrix& B, int row_start, int col_start, int ops_num, Matrix& result){
    for(int i = 0; i < ops_num; i++){
        int row = row_start + (col_start + i) / B.getCols();
        int col = (col_start + i) % B.getCols();
        float sum = 0.0;

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

Matrix Matrix::operator+(float value) const{
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

Matrix Matrix::operator-(float value) const{
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

Matrix Matrix::operator*(float value) const{
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

Matrix Matrix::operator/(float value) const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            result.set(r, c, get(r, c) / value);
        }
    }

    return result;
}

Matrix Matrix::pow(float power) const{
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
            float sum = 0.0;

            for(int r = 0; r < rows; r++){
                sum += get(r, c);
            }

            result.set(0, c, sum);
        }

        return result;
    }else if(dim == 1){
        Matrix result(rows, 1);

        for(int r = 0; r < rows; r++){
            float sum = 0.0;

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
            float max = get(0, c);

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
            float max = get(r, 0);

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
            float max = get(0, c);
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
            float max = get(r, 0);
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

Matrix Matrix::std(int dim) const{
    if(dim == 0){
        Matrix mean = this->mean(0);
        Matrix result(1, cols);

        for(int c = 0; c < cols; c++){
            float sum = 0.0;

            for(int r = 0; r < rows; r++){
                sum += std::pow(get(r, c) - mean.get(0, c), 2);
            }

            result.set(0, c, sqrt(sum / rows));
        }

        return result;
    }else if(dim == 1){
        Matrix mean = this->mean(1);
        Matrix result(rows, 1);

        for(int r = 0; r < rows; r++){
            float sum = 0.0;

            for(int c = 0; c < cols; c++){
                sum += std::pow(get(r, c) - mean.get(r, 0), 2);
            }

            result.set(r, 0, sqrt(sum / cols));
        }

        return result;
    }else{
        throw invalid_argument("Std dim must be 0 or 1");
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

Matrix Matrix::relu() const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            if(get(r, c) > 0){
                result.set(r, c, get(r, c));
            }else{
                result.set(r, c, 0.0);
            }
        }
    }

    return result;
}

Matrix Matrix::dRelu() const{
    Matrix result(rows, cols);

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            if(get(r, c) > 0){
                result.set(r, c, 1.0);
            }else{
                result.set(r, c, 0.0);
            }
        }
    }

    return result;
}

Matrix Matrix::softmax() const{
    // Subtracting max for numerical stability
    Matrix exp_values_minus_max = (*this - max(1)).exp();
    return exp_values_minus_max / exp_values_minus_max.sum(1);
}

// Normalize each row to have mean 0 and std 1
Matrix Matrix::normalize() const{
    return (*this - mean(1)) / std(1);
}
