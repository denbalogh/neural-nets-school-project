#include <iostream>
#include <chrono>
#include "matrix/matrix.h"
#include "data_loader/data_loader.h"
#include "batch/batch.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if(argc == 2 && string(argv[1]) == "aura"){
         // Aura has 128 cores
        Matrix::setMaxThreads(128);
    } else {
        // My laptop has 6 cores
        Matrix::setMaxThreads(6);
    }

    DataLoader test_loader = DataLoader(TEST);

    Batch batch = test_loader.getBatch();

    cout << "Batch shape:" << endl;
    cout << batch.getData().getShape() << endl;

    return 0;
}
