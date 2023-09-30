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

    cout << "Data loaded" << endl;

    Batch batch = loader.getBatch();
    Matrix x = batch.getData();
    vector<int> y = batch.getLabels();

    cout << "Batch loaded" << endl;

    int hidden_size = 32;

    Matrix W1 = Matrix(ITEM_SIZE, hidden_size, RAND);
    Matrix b1 = Matrix(1, hidden_size, RAND);

    Matrix W2 = Matrix(hidden_size, 10, RAND);
    Matrix b2 = Matrix(1, 10, RAND);

    Matrix l1 = x.matmul(W1) + b1;
    l1 = l1.relu();

    cout << "Layer 1 output: " << endl;
    l1.printValues();

    Matrix l2 = l1.matmul(W2) + b2;

    l2 = l2.softmax();

    cout << "Layer 2 output: " << endl;
    l2.printValues();

    double loss = crossEntropy(l2, y);

    cout << "Loss: " << loss << endl;

    return 0;
}
