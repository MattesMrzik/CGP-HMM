main:  Viterbi.o
	g++ -pthread Viterbi.o -o Viterbi -std=c++11
Viterbi.o: Viterbi.cc json.hpp
	g++ -c Viterbi.cc -std=c++11
clean :
	rm *.o Viterbi
