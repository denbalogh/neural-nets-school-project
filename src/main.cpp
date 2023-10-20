#include <iostream>
#include <chrono>
#include "matrix/matrix.h"
#include "data_loader/data_loader.h"
#include "batch/batch.h"
#include "loss/loss.h"
#include "layer/layer.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if(argc == 2 && string(argv[1]) == "aura"){
         // Aura has 128 cores
        Matrix::setMaxThreads(128);
    } else {
        // My laptop has 6 cores
        Matrix::setMaxThreads(6);
    }

    DataLoader loader = DataLoader(TRAIN, 0.1);

    int hiddenSize = 512;
    int batchSize = 512;
    int iterations = 1000;

    double lr = 0.01, prevValAcc = 0.0;

    Layer layer1 = Layer(ITEM_SIZE, hiddenSize, "tanh");
    Layer layer2 = Layer(hiddenSize, 10, "softmax");

    Batch valData = loader.getValData();
    Matrix valX = valData.getX();
    vector<int> valY = valData.getY();

    for(int i = 0; i < iterations; i++){
        Batch batch = loader.getTrainBatch(batchSize);
        Matrix x = batch.getX();
        vector<int> y = batch.getY();

        // Forward pass
        Matrix h = layer1.forward(x);
        Matrix logits = layer2.forward(h);
        double loss = crossEntropy(logits, y);

        cout << "i: " << i << ", train loss: " << loss << endl;

        if(i != 0 && i % 20 == 0){
            layer1.setTrain(false);
            layer2.setTrain(false);
            Matrix valH = layer1.forward(valX);
            Matrix valLogits = layer2.forward(valH);
            double valAcc = accuracy(valLogits, valY);
            cout << "------- Val accuracy: " << valAcc << endl;
            layer1.setTrain(true);
            layer2.setTrain(true);

            if(valAcc < prevValAcc){
                lr *= 0.5;
                cout << "------- Learning rate: " << lr << endl;
            }

            prevValAcc = valAcc;
        }

        //Backward pass
        Matrix dh = layer2.backward(h, y);
        layer1.backward(x, dh);

        // Update weights
        layer1.update(lr);
        layer2.update(lr);
    }

    return 0;
}
