#include <iostream>
#include <chrono>
#include "matrix/matrix.h"
#include "data_loader/data_loader.h"
#include "batch/batch.h"
#include "loss/loss.h"
#include "layer/layer.h"
#include "MLP/MLP.h"
#include "utils/utils.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if(argc == 2 && string(argv[1]) == "aura"){
         // Aura has 256 threads
        Matrix::setMaxThreads(256);
    } else {
        // My laptop has 6 threads
        Matrix::setMaxThreads(6);
    }

    DataLoader trainLoader = DataLoader(TRAIN, 0.1), testLoader = DataLoader(TEST);

    // Hyperparameters
    int batchSize = 256;
    int hiddenSize = 1024;
    int nHiddenLayers = 4;

    // Training parameters
    int iterations = 1000;
    double lr = 0.001;

    // Helper variables
    double prevValAcc = 0.0;
    int prevValLessCount = 0;

    MLP network = MLP(ITEM_SIZE, hiddenSize, nHiddenLayers, 10, "relu", "softmax");

    Batch valData = trainLoader.getValData();
    Matrix valX = valData.getX().normalize();
    vector<int> valY = valData.getY();

    for(int i = 0; i < iterations; i++){
        Batch batch = trainLoader.getTrainBatch(batchSize);
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

        if(prevValLessCount >= 2){
            cout << "------- Early stopping" << endl;
            break;
        }

        //Backward pass
        network.backward(x, y);

        // Update weights
        network.update(lr);
    }

    // Final evaluation and predictions saving
    Batch testData = testLoader.getAllData();
    Matrix testX = testData.getX().normalize();
    vector<int> testY = testData.getY();

    network.setTrain(false);
    Matrix testPredictions = network.forward(testX);

    // Test set evaluation
    double testAcc = accuracy(testPredictions, testY);
    cout << "------- Test accuracy: " << testAcc << endl;

    Batch trainData = trainLoader.getAllData();
    Matrix trainX = trainData.getX().normalize();
    Matrix trainPredictions = network.forward(trainX); 

    cout << "------- Saving predictions......." << endl;
    // Save predictions
    savePredictions(trainPredictions, "../train_predictions.csv");
    savePredictions(testPredictions, "../test_predictions.csv");

    return 0;
}
