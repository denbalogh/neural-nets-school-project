#include <iostream>
#include <chrono>
#include "matrix/matrix.h"
#include "data_loader/data_loader.h"
#include "batch/batch.h"
#include "loss/loss.h"
#include "layer/layer.h"
#include "MLP/MLP.h"

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

    // Hyperparameters
    int batchSize = 256;
    int hiddenSize = 64;
    int nHiddenLayers = 3;

    // Training parameters
    int iterations = 1000;
    double lr = 0.001;

    // Helper variables
    double prevValAcc = 0.0;
    int prevValLessCount = 0;

    MLP network = MLP(ITEM_SIZE, hiddenSize, nHiddenLayers, 10, "tanh", "softmax");

    Batch valData = loader.getValData();
    Matrix valX = valData.getX().normalize();
    vector<int> valY = valData.getY();

    for(int i = 0; i < iterations; i++){
        Batch batch = loader.getTrainBatch(batchSize);
        Matrix x = batch.getX().normalize();
        vector<int> y = batch.getY();

        // Forward pass
        Matrix logits = network.forward(x);
        double loss = crossEntropy(logits, y);

        cout << "i: " << i << ", train loss: " << loss << endl;

        if(i != 0 && i % 20 == 0){
            network.setTrain(false);

            Matrix valLogits = network.forward(valX);
            double valAcc = accuracy(valLogits, valY);
            cout << "------- Val accuracy: " << valAcc << endl;

            network.setTrain(true);

            if(valAcc <= prevValAcc){
                lr *= 0.5;
                cout << "------- Learning rate: " << lr << endl;
                prevValLessCount++;
            } else {
                prevValLessCount = 0;
            }

            prevValAcc = valAcc;
        }

        if(prevValLessCount >= 3){
            cout << "------- Early stopping" << endl;
            break;
        }

        //Backward pass
        network.backward(x, y);

        // Update weights
        network.update(lr);
    }

    return 0;
}
