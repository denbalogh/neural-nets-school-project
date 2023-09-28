#ifndef BATCH_H
#define BATCH_H

#include <iostream>
#include <vector>
#include <string>

#include "../matrix/matrix.h"

using namespace std;

class Batch {
    private:
        Matrix data;
        vector<int> labels;
    
    public:
        Batch(Matrix data, vector<int> labels);
        Matrix getData();
        vector<int> getLabels();
};

#endif
