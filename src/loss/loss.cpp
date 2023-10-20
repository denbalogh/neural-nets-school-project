#include "loss.h"

double accuracy(const Matrix& y, const vector<int>& y_hat) {
    Matrix y_argmax = y.argmax(1);
    int correct = 0;

    for(int r = 0; r < y.getRows(); r++) {
        if(y_hat[r] == y_argmax.get(r, 0)) {
            correct++;
        }
    }

    return (double) correct / y.getRows();
}

double crossEntropy(const Matrix& y, const vector<int>& y_hat) {
    Matrix loss(y.getRows(), 1, ZEROS);

    for(int r = 0; r < y.getRows(); r++) {
        for(int c = 0; c < y.getCols(); c++) {
            if(y_hat[r] == c) {
                loss.set(r, 0, -log(y.get(r, c)));
            }
        }
    }

    return loss.mean(0).get(0, 0);
}

Matrix crossEntropyGrad(const Matrix& logits, const vector<int>& y_hat) {
    Matrix dlogits = logits.softmax();

    for(int r = 0; r < dlogits.getRows(); r++) {
        for(int c = 0; c < dlogits.getCols(); c++) {
            if(y_hat[r] == c) {
                dlogits.set(r, c, dlogits.get(r, c) - 1);
            }
        }
    }

    return dlogits / y_hat.size();
}
