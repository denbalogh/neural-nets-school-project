#ifndef BATCH_H
#define BATCH_H

#include <iostream>
#include <vector>
#include <string>

#include "../matrix/matrix.h"

using namespace std;

class Batch {
    private:
        Matrix X;
        vector<int> Y;
    
    public:
        Batch(Matrix X, vector<int> Y);
        Matrix& getX();
        vector<int>& getY();
};

#endif
