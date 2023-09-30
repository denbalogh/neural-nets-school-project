#include "loss.h"

double crossEntropy(Matrix& y, vector<int> y_hat) {
    Matrix loss(y.getRows(), 1, ZEROS);

    for(int r = 0; r < y.getRows(); r++) {
        for(int c = 0; c < y.getCols(); c++) {
            if(y_hat[r] == c) {
                loss.setValue(r, 0, -log(y.getValue(r, c)));
            }
        }
    }

    return loss.mean(0).getValue(0, 0);
}
