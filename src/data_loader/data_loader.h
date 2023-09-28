#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

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
    
    public:
        DataLoader(DataType type);
        Batch getBatch();
};

#endif
