#include "data_loader.h"

DataLoader::DataLoader(DataType type) {
    string filenameX, filenameY;

    if(type == TEST) {
        filenameX = "../data/fashion_mnist_test_vectors.csv";
        filenameY = "../data/fashion_mnist_test_labels.csv";
    } else {
        filenameX = "../data/fashion_mnist_train_vectors.csv";
        filenameY = "../data/fashion_mnist_train_labels.csv";
    }

    ifstream fileX(filenameX), fileY(filenameY);
    string line;

    if(fileX.is_open()) {
        while(getline(fileX, line)) {
            dataX.push_back(line);
        }
    } else {
        throw invalid_argument("File for X not found");
    }

    if(fileY.is_open()) {
        while(getline(fileY, line)) {
            dataY.push_back(stoi(line));
        }
    } else {
        throw invalid_argument("File for Y not found");
    }

    fileX.close();
    fileY.close();
}

Batch DataLoader::getBatch() {
    vector<string> batchX;
    vector<int> batchY;

    for(int i = 0; i < BATCH_SIZE; i++) {
        int index = rand() % dataX.size();
        batchX.push_back(dataX[index]);
        batchY.push_back(dataY[index]);
    }

    Matrix data = Matrix(BATCH_SIZE, ITEM_SIZE, ZEROS);
    vector<int> labels = batchY;

    for(int i = 0; i < BATCH_SIZE; i++) {
        string line = batchX[i];
        int j = 0;
        stringstream ssin(line);
        while(ssin.good() && j < ITEM_SIZE) {
            string value;
            getline(ssin, value, ',');
            data.set(i, j, stod(value));
            j++;
        }
    }

    return Batch(data, labels);
}
