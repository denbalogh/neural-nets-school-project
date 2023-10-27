#include "data_loader.h"

DataLoader::DataLoader(DataType type, float val_split) {
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

    if(val_split > 0.0) {
        int valCount = val_split * TRAIN_ITEMS_COUNT;
        valStartIndex = TRAIN_ITEMS_COUNT - valCount;
    }
}

Batch DataLoader::getTrainBatch(int batchSize) const{
    vector<string> batchX;
    vector<int> batchY;

    for(int i = 0; i < batchSize; i++) {
        int index = rand() % valStartIndex;
        batchX.push_back(dataX[index]);
        batchY.push_back(dataY[index]);
    }

    Matrix data = Matrix(batchSize, ITEM_SIZE, ZEROS);
    vector<int> labels = batchY;

    for(int i = 0; i < batchSize; i++) {
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

Batch DataLoader::getValData() const{
    #ifdef DEBUG
        if(valStartIndex == TRAIN_ITEMS_COUNT - 1) {
            cout << "No validation data" << endl;
            exit(1);
        }
    #endif

    vector<string> batchX;
    vector<int> batchY;

    for(int i = valStartIndex; i < TRAIN_ITEMS_COUNT; i++) {
        batchX.push_back(dataX[i]);
        batchY.push_back(dataY[i]);
    }

    Matrix data = Matrix(batchX.size(), ITEM_SIZE, ZEROS);
    vector<int> labels = batchY;

    for(vector<string>::size_type i = 0; i < batchX.size(); i++) {
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