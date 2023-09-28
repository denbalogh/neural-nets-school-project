FLAGS = -std=c++17 -Wall -pedantic

network: main.o matrix.o data_loader.o batch.o
	g++ $(FLAGS) main.o matrix.o data_loader.o batch.o -o network

main.o: main.cpp
	g++ $(FLAGS) -c main.cpp

matrix.o: matrix/matrix.cpp matrix/matrix.h
	g++ $(FLAGS) -c matrix/matrix.cpp

data_loader.o: data_loader/data_loader.cpp data_loader/data_loader.h
	g++ $(FLAGS) -c data_loader/data_loader.cpp

batch.o: batch/batch.cpp batch/batch.h
	g++ $(FLAGS) -c batch/batch.cpp

clean:
	rm *.o network
