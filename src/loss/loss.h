#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include <vector>
#include <cmath>

#include "../debug.h"
#include "../matrix/matrix.h"

using namespace std;

float accuracy(const Matrix& y, const vector<int>& y_hat);
float crossEntropy(const Matrix& y, const vector<int>& y_hat);
Matrix crossEntropyGrad(const Matrix& probs, const vector<int>& y_hat);

#endif
