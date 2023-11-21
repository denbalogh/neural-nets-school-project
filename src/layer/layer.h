#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <cmath>

#include "../debug.h"
#include "../matrix/matrix.h"
#include "../loss/loss.h"

using namespace std;

#define RO 0.9
#define SIGMA 1e-8

class Layer {
    private:
        Matrix W;
        Matrix b;
        string activation;
        Matrix hpreact;
        Matrix h;
        Matrix dW;
        Matrix db;
        Matrix rW;
        Matrix rb;
        bool train = true;
    
    public:
        Layer(int fin, int fout, string activation);
        Matrix forward(const Matrix& x);
        Matrix backward(const Matrix& x, const vector<int>& y_hat); // softmax layer
        Matrix backward(const Matrix& x, const Matrix& dh); // hidden layer
        void update(float lr);
        void setTrain(bool train);
        Matrix getH() const { return h; }
};

#endif
