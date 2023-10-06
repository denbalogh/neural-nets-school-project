#include <iostream>
#include <chrono>
#include "matrix/matrix.h"
#include "data_loader/data_loader.h"
#include "batch/batch.h"
#include "loss/loss.h"

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

    Matrix W1 = Matrix(ITEM_SIZE, hidden_size, RAND);
    Matrix b1 = Matrix(1, hidden_size, RAND);

    Matrix W2 = Matrix(hidden_size, 10, RAND);
    Matrix b2 = Matrix(1, 10, RAND);

    for(int i = 0; i < 1000; i++){
        Batch batch = loader.getBatch();
        Matrix x = batch.getData();
        vector<int> y = batch.getLabels();

        // Forward pass
        Matrix hpreact = x.matmul(W1) + b1;
        Matrix h = hpreact.tanh();
        Matrix logits = h.matmul(W2) + b2;
        Matrix probs = logits.softmax();
        double loss = crossEntropy(probs, y);

        cout << "Loss: " << loss << endl;

        //Backward pass
        Matrix dlogits = crossEntropyGrad(logits, y);

        Matrix W2T = W2.transpose();
        Matrix dh = dlogits.matmul(W2T);

        Matrix dW2 = h.transpose().matmul(dlogits);
        Matrix db2 = dlogits.sum(0);

        Matrix hAsOnes = Matrix(h.getRows(), h.getCols(), ONES);
        Matrix hPow2 = h.pow(2);
        Matrix hAsOnesMinusHPow2 = hAsOnes - hPow2;
        Matrix dhpreact = dh * hAsOnesMinusHPow2;

        Matrix dW1 = x.transpose().matmul(dhpreact);
        Matrix db1 = dhpreact.sum(0);

        // Update weights
        dW1 = dW1 * lr;
        db1 = db1 * lr;
        dW2 = dW2 * lr;
        db2 = db2 * lr;

        W1 = W1 - dW1;
        b1 = b1 - db1;
        W2 = W2 - dW2;
        b2 = b2 - db2;
    }

    return 0;
}
