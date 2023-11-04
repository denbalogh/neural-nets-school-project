#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <vector>
#include <string>

#include "../layer/layer.h"

using namespace std;

class MLP {
    private:
        vector<Layer> layers;

    public:
        MLP(int fin, int hiddenSize, int nHiddenLayers, int fout, string hiddenActivation, string outputActivation);
        void setTrain(bool train);
        Matrix forward(const Matrix& x);
        void backward(const Matrix& x, const vector<int>& y_hat);
        void update(double lr);
};

#endif
