#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <cmath>

#include "../matrix/matrix.h"
#include "../loss/loss.h"

using namespace std;

class Layer {
    private:
        Matrix W;
        Matrix b;
        string activation;
        Matrix hpreact;
        Matrix h;
        Matrix dW;
        Matrix db;
    
    public:
        Layer(int fin, int fout, string activation);
        Matrix forward(Matrix& x);
        Matrix backward(Matrix& x, vector<int> y_hat); // softmax layer
        Matrix backward(Matrix& x, Matrix& dh); // hidden layer
        void update(double lr);
};

#endif
