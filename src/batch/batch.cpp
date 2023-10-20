#include "batch.h"

Batch::Batch(Matrix X, vector<int> Y) : X(X), Y(Y) {}

Matrix& Batch::getX(){
    return X;
}

vector<int>& Batch::getY(){
    return Y;
}
