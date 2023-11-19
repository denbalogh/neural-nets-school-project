#include <fstream>

#include "utils.h"

void savePredictions(const Matrix& predictions, string filename) {
    ofstream file;
    file.open(filename);

    if(!file.is_open()) {
        cout << "Error opening output file: " << filename << endl;
        exit(1);
    }

    Matrix predictionsMax = predictions.argmax(1);

    for(int i = 0; i < predictionsMax.getRows(); i++) {
        file << predictionsMax.get(i, 0) << endl;
    }

    file.close();
}
