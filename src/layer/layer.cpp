#include "layer.h"

Layer::Layer(int fin, int fout, string activation) {
    W = Matrix(fin, fout, RAND);
    b = Matrix(1, fout, RAND);
    this->activation = activation;
}

Matrix Layer::forward(Matrix& x) {
    hpreact = x.matmul(W) + b;
    
    if(activation == "tanh"){
        h = hpreact.tanh();
    } else if(activation == "softmax"){
        h = hpreact.softmax();
    } else {
        cout << "Invalid activation function" << endl;
        exit(1);
    }

    return h;
}

Matrix Layer::backward(Matrix& x, vector<int> y_hat){
    if(activation != "softmax"){
        cout << "Need to be softmax activation" << endl;
        exit(1);
    }

    Matrix dlogits = crossEntropyGrad(hpreact, y_hat);

    dW = x.transpose().matmul(dlogits);
    db = dlogits.sum(0);

    Matrix WT = W.transpose();
    return dlogits.matmul(WT);
}

Matrix Layer::backward(Matrix& x, Matrix& dh){
    if(activation != "tanh"){
        cout << "Need to be tanh activation" << endl;
        exit(1);
    }

    Matrix hAsOnes = Matrix(h.getRows(), h.getCols(), ONES);
    Matrix hPow2 = h.pow(2);

    Matrix hAsOnesMinusHPow2 = hAsOnes - hPow2;
    Matrix dhpreact = dh * hAsOnesMinusHPow2;

    dW = x.transpose().matmul(dhpreact);
    db = dhpreact.sum(0);

    Matrix WT = W.transpose();
    return dhpreact.matmul(WT);
}

void Layer::update(double lr){
    dW = dW * lr;
    db = db * lr;

    W = W - dW;
    b = b - db;
}
