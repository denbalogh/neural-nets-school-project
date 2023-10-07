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

    DataLoader loader = DataLoader(TRAIN);

    int hidden_size = 32;
    double lr = 0.01;
    int iterations = 1000;

    Layer layer1 = Layer(ITEM_SIZE, hidden_size, "tanh");
    Layer layer2 = Layer(hidden_size, 10, "softmax");

    for(int i = 0; i < iterations; i++){
        Batch batch = loader.getBatch();
        Matrix x = batch.getData();
        vector<int> y = batch.getLabels();

        // Forward pass
        Matrix h = layer1.forward(x);
        Matrix logits = layer2.forward(h);
        double loss = crossEntropy(logits, y);

        cout << "Loss: " << loss << endl;

        //Backward pass
        Matrix dh = layer2.backward(h, y);
        layer1.backward(x, dh);

        // Update weights
        layer1.update(lr);
        layer2.update(lr);
    }

    return 0;
}
