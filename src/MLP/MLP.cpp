#include "MLP.h"

MLP::MLP(int fin, int hiddenSize, int nHiddenLayers, int fout, string hiddenActivation, string outputActivation) {
    layers.push_back(Layer(fin, hiddenSize, hiddenActivation));

    for(int i = 0; i < nHiddenLayers - 2; i++) {
        layers.push_back(Layer(hiddenSize, hiddenSize, hiddenActivation));
    }
    
    layers.push_back(Layer(hiddenSize, fout, outputActivation));
}

void MLP::setTrain(bool train) {
    for(int i = 0; i < (int)layers.size(); i++) {
        layers[i].setTrain(train);
    }
}

Matrix MLP::forward(const Matrix& x) {
    Matrix h = x;

    for(int i = 0; i < (int)layers.size(); i++) {
        h = layers[i].forward(h);
    }

    return h;
}

void MLP::backward(const Matrix& x, const vector<int>& y_hat) {
    Matrix dh = layers[layers.size() - 1].backward(layers[layers.size() - 2].getH(), y_hat);
    for(int i = (int)layers.size() - 2; i >= 0; i--) {
        if(i > 0){
            dh = layers[i].backward(layers[i - 1].getH(), dh);
        } else {
            layers[i].backward(x, dh);
        }
    }
}

void MLP::update(double lr) {
    for(int i = 0; i < (int)layers.size(); i++) {
        layers[i].update(lr);
    }
}
