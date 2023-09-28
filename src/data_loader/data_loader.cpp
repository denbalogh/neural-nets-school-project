#include "data_loader.h"

DataLoader::DataLoader(string filenameX, string filenameY) {
    ifstream fileX(filenameX), fileY(filenameY);
    string line;

    if(fileX.is_open()) {
        while(getline(fileX, line)) {
            dataX.push_back(line);
        }

        // Get the number of items in each data row
        itemSize = count(dataX[0].begin(), dataX[0].end(), ',') + 1;
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

int DataLoader::getNumberOfItems() {
    return dataX.size();
}

int DataLoader::getItemSize() {
    return itemSize;
}

Batch DataLoader::getBatch(int batchSize) {
    if(batchSize > dataX.size()) {
        throw invalid_argument("Batch size is larger than the number of items");
    }

    vector<string> batchX;
    vector<int> batchY;

    for(int i = 0; i < batchSize; i++) {
        int index = rand() % dataX.size();
        batchX.push_back(dataX[index]);
        batchY.push_back(dataY[index]);
    }

    Matrix data = Matrix(batchSize, itemSize, ZEROS);
    vector<int> labels = batchY;

    for(int i = 0; i < batchSize; i++) {
        string line = batchX[i];
        int j = 0;
        stringstream ssin(line);
        while(ssin.good() && j < itemSize) {
            string value;
            getline(ssin, value, ',');
            data.setValue(i, j, stod(value));
            j++;
        }
    }

    return Batch(data, labels);
}
