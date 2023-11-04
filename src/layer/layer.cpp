#include "layer.h"

Layer::Layer(int fin, int fout, string activation): 
W(Matrix(fin, fout, 0, 1.0 / sqrt(fin))), 
b(Matrix(1, fout, ZEROS)),
activation(activation),
rW(Matrix(fin, fout, ONES)),
rb(Matrix(1, fout, ONES)) {}

void Layer::setTrain(bool train){
    this->train = train;
}

Matrix Layer::forward(const Matrix& x) {
    Matrix hpreact, h;
    hpreact = x.matmul(W) + b;
    
    if(activation == "tanh"){
        h = hpreact.tanh();
    } else if(activation == "relu"){
        h = hpreact.relu();
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
        if(activation != "tanh" && activation != "relu"){
            cout << "Need to be tanh or relu activation" << endl;
            exit(1);
        }
    #endif

    Matrix dhpreact;

    if(activation == "tanh"){
        dhpreact = dh * h.dTanh();
    } else {
        //Relu
        dhpreact = dh * h.dRelu();
    }

    dW = x.transpose().matmul(dhpreact);
    db = dhpreact.sum(0);

    Matrix WT = W.transpose();

    return dhpreact.matmul(WT);
}

void Layer::update(double lr){
    // RMSProp method
    rW = rW * RO + dW.pow(2) * (1 - RO);
    dW = (rW + SIGMA).pow(-0.5) * dW * lr;

    rb = rb * RO + db.pow(2) * (1 - RO);
    db = (rb + SIGMA).pow(-0.5) * db * lr;

    W = W - dW;
    b = b - db;
}
