FLAGS = -std=c++17 -Wall -pedantic -O3

network: main.o matrix.o data_loader.o batch/batch.h loss.o layer.o MLP.o utils.o
	g++ $(FLAGS) main.o matrix.o data_loader.o loss.o layer.o MLP.o utils.o -o network

main.o: main.cpp
	g++ $(FLAGS) -c main.cpp

matrix.o: matrix/matrix.cpp matrix/matrix.h debug.h
	g++ $(FLAGS) -c matrix/matrix.cpp

data_loader.o: data_loader/data_loader.cpp data_loader/data_loader.h debug.h
	g++ $(FLAGS) -c data_loader/data_loader.cpp

loss.o: loss/loss.cpp loss/loss.h debug.h
	g++ $(FLAGS) -c loss/loss.cpp

layer.o: layer/layer.cpp layer/layer.h debug.h
	g++ $(FLAGS) -c layer/layer.cpp

MLP.o: MLP/MLP.cpp MLP/MLP.h debug.h
	g++ $(FLAGS) -c MLP/MLP.cpp

utils.o: utils/utils.cpp utils/utils.h
	g++ $(FLAGS) -c utils/utils.cpp

clean:
	rm *.o network
