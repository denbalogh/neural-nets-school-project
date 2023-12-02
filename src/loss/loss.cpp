#include "loss.h"

float accuracy(const Matrix& y, const vector<int>& y_hat) {
    Matrix y_argmax = y.argmax(1);
    int correct = 0;

    for(int r = 0; r < y.getRows(); r++) {
        if(y_hat[r] == y_argmax.get(r, 0)) {
            correct++;
        }
    }

    return (float) correct / y.getRows();
}

float crossEntropy(const Matrix& y, const vector<int>& y_hat) {
    #ifdef DEBUG
        if(y.getRows() != y_hat.size()) {
            cout << "CrossEntropy: invalid dimensions - " << y.getRows() << ", " << y_hat.size() << endl;
            exit(1);
        }
    #endif

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

Matrix crossEntropyGrad(const Matrix& probs, const vector<int>& y_hat) {
    #ifdef DEBUG
        if(probs.getRows() != y_hat.size()) {
            cout << "CrossEntropyGrad: invalid dimensions - " << probs.getRows() << ", " << y_hat.size() << endl;
            exit(1);
        }
    #endif

    // This expression for gradient was derived based on this video from Andrej Karpathy:
    // https://youtu.be/q8SA3rM6ckI?t=5317

    Matrix dlogits = probs.clone();

    for(int r = 0; r < dlogits.getRows(); r++) {
        for(int c = 0; c < dlogits.getCols(); c++) {
            if(y_hat[r] == c) {
                dlogits.set(r, c, dlogits.get(r, c) - 1);
            }
        }
    }

    return dlogits / y_hat.size();
}
