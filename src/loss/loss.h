#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include <vector>
#include <cmath>

#include "../matrix/matrix.h"

using namespace std;

double crossEntropy(Matrix& y, vector<int> y_hat);

#endif
