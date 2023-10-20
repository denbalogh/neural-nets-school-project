#include "batch.h"

Batch::Batch(Matrix x, vector<int> y) {
    data = x;
    labels = y;
}

Matrix& Batch::getData(){
    return data;
}

vector<int>& Batch::getLabels(){
    return labels;
}
