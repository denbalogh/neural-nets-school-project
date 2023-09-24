#include <iostream>
#include "matrix/matrix.h"

using namespace std;

int main() {
    Matrix W(10, 10, RAND);
    Matrix x(10, 10, RAND);

    Matrix sum = W + x;

    W.printValues();
    x.printValues();

    cout << "Sum:" << endl;

    sum.printValues();

    return 0;
}
