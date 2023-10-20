#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include <vector>
#include <cmath>

#include "../matrix/matrix.h"

using namespace std;

double crossEntropy(const Matrix& y, const vector<int>& y_hat);
Matrix crossEntropyGrad(const Matrix& logits, const vector<int>& y_hat);

#endif
