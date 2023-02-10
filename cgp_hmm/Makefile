main:  Viterbi.o
	g++ Viterbi.o -o Viterbi -std=c++11
Viterbi.o: Viterbi.cc json.hpp
	g++ -c Viterbi.cc
clean :
	rm *.o main

	# main: main.o sort.o
	# 	g++ main.o sort.o -o main
	# main.o: main.cpp sort.h
	# 	g++ -c main.cpp
	# sort.o: sort.cpp sort.h
	# 	g++ -c sort.cpp
	# clean:
	# 	rm *.o main
