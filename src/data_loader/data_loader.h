#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "../debug.h"
#include "../matrix/matrix.h"
#include "../batch/batch.h"

#define ITEM_SIZE 784
#define TRAIN_ITEMS_COUNT 60000
#define TEST_ITEMS_COUNT 10000

enum DataType {
    TRAIN,
    TEST
};

using namespace std;

class DataLoader {
    private:
        vector<string> dataX;
        vector<int> dataY;
        int valStartIndex = TRAIN_ITEMS_COUNT - 1;
    
    public:
        DataLoader(DataType type, float val_split = 0.0);
        Batch getTrainBatch(int batchSize) const;
        Batch getValData() const;
};

#endif
