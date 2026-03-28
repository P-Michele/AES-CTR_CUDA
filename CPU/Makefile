CXX = g++
CXXFLAGS = -Wall -Wextra -O3

main: main.o aes.o
	$(CXX) $(CXXFLAGS) -o main main.o aes.o

main.o: main.cpp aes.h
	$(CXX) $(CXXFLAGS) -c main.cpp

aes.o: aes.cpp aes.h box.h
	$(CXX) $(CXXFLAGS) -c aes.cpp

clean:
	rm -f *.o main