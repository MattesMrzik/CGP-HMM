LDLIBS = -lboost_program_options
LINKSHARED = -L/usr/local/lib/libboost_program_options.so

main:  Viterbi.o
	g++ -pthread Viterbi.o -o Viterbi -std=c++11 $(LINKSHARED)  $(LDLIBS)
	# g++ -pthread -lboost_program_options Viterbi.o -o Viterbi -std=c++11 $(LDLIBS)
Viterbi.o: Viterbi.cc json.hpp
	g++ -c Viterbi.cc -std=c++11
clean :
	rm *.o Viterbi
