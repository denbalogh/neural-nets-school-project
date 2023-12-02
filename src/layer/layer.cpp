#include "layer.h"

Layer::Layer(int fin, int fout, string activation): 
// Properly initialize weights and biases
W(Matrix(fin, fout, 0, sqrt(2) / sqrt(activation == "relu" ? fin : fin + fout))), 
b(Matrix(1, fout, ZEROS)),
activation(activation),
// RMSProp variables
rW(Matrix(fin, fout, ONES)),
rb(Matrix(1, fout, ONES)) {}

void Layer::setTrain(bool train){
    this->train = train;
}

Matrix Layer::forward(const Matrix& x) {
    Matrix hpreact, h;
    hpreact = x.matmul(W) + b;
    
    if(activation == "relu"){
        h = hpreact.relu();
    } else if(activation == "softmax"){
        h = hpreact.softmax();
    } else {
        cout << "Invalid activation function" << endl;
        exit(1);
    }

    // Save these values for backpropagation during training
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

    // These expressions for gradients were derived based on this video from Andrej Karpathy:
    // https://youtu.be/q8SA3rM6ckI?t=2505

    dW = x.transpose().matmul(dlogits);
    db = dlogits.sum(0);

    return dlogits.matmul(W.transpose());
}

Matrix Layer::backward(const Matrix& x, const Matrix& dh){
    #ifdef DEBUG
        if(activation != "relu"){
            cout << "Need to be relu activation" << endl;
            exit(1);
        }
    #endif

    Matrix dhpreact = dh * h.dRelu();

    // These expressions for gradients were derived based on this video from Andrej Karpathy:
    // https://youtu.be/q8SA3rM6ckI?t=2505

    dW = x.transpose().matmul(dhpreact);
    db = dhpreact.sum(0);

    return dhpreact.matmul(W.transpose());
}

void Layer::update(float lr){
    // RMSProp method
    rW = rW * RO + dW.pow(2) * (1 - RO);
    dW = (rW + SIGMA).pow(-0.5) * dW * lr;

    rb = rb * RO + db.pow(2) * (1 - RO);
    db = (rb + SIGMA).pow(-0.5) * db * lr;

    W = W - dW;
    b = b - db;
}
