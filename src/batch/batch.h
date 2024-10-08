#ifndef BATCH_H
#define BATCH_H

#include "../matrix/matrix.h"

#include <vector>

using namespace std;

class Batch {
    private:
        Matrix X;
        vector<int> Y;
    
    public:
        Batch(){};
        Batch(Matrix X, vector<int> Y): X(X), Y(Y) {};
        Matrix& getX() { return X; };
        vector<int>& getY() { return Y; };
};

#endif
