#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "../matrix/matrix.h"
#include "../batch/batch.h"

using namespace std;

class DataLoader {
    private:
        vector<string> dataX;
        vector<int> dataY;
        int itemSize;
    
    public:
        DataLoader(string filenameX, string filenameY);
        int getNumberOfItems();
        int getItemSize();
        Batch getBatch(int batchSize);
};

#endif
