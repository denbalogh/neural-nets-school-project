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
        Batch(Matrix X, vector<int> Y): X(X), Y(Y) {};
        Matrix& getX() { return X; };
        vector<int>& getY() { return Y; };
};

#endif
