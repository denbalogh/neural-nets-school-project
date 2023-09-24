FLAGS = -std=c++17 -Wall -pedantic

network: main.o matrix.o
	g++ $(FLAGS) main.o matrix.o -o network

main.o: main.cpp
	g++ $(FLAGS) -c main.cpp

matrix.o: matrix/matrix.cpp matrix/matrix.h
	g++ $(FLAGS) -c matrix/matrix.cpp

clean:
	rm *.o network
