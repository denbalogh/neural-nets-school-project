#include "layer.h"

Layer::Layer(int fin, int fout, string activation): W(Matrix(fin, fout, RAND)), b(Matrix(1, fout, RAND)), activation(activation) {}

void Layer::setTrain(bool train){
    this->train = train;
}

Matrix Layer::forward(const Matrix& x) {
    Matrix hpreact, h;
    hpreact = x.matmul(W) + b;
    
    if(activation == "tanh"){
        h = hpreact.tanh();
    } else if(activation == "softmax"){
        h = hpreact.softmax();
    } else {
        cout << "Invalid activation function" << endl;
        exit(1);
    }

    if(train){
        this->hpreact = hpreact;
        this->h = h;
    }

    return h;
}

Matrix Layer::backward(const Matrix& x, const vector<int>& y_hat){
    #ifdef DEBUG
        if(activation != "softmax"){
            cout << "Need to be softmax activation" << endl;
            exit(1);
        }
    #endif

    Matrix dlogits = crossEntropyGrad(h, y_hat);

    dW = x.transpose().matmul(dlogits);
    db = dlogits.sum(0);

    Matrix WT = W.transpose();

    return dlogits.matmul(WT);
}

Matrix Layer::backward(const Matrix& x, const Matrix& dh){
    #ifdef DEBUG
        if(activation != "tanh"){
            cout << "Need to be tanh activation" << endl;
            exit(1);
        }
    #endif

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
