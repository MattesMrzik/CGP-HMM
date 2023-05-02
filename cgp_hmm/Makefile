LDLIBS = -lboost_program_options
LINKSHARED = -L/usr/local/lib/libboost_program_options.so

main:  Viterbi.o
	g++ -pthread Viterbi.o -o Viterbi -std=c++17 $(LINKSHARED)  $(LDLIBS)
	# g++ -pthread -lboost_program_options Viterbi.o -o Viterbi -std=c++17 $(LDLIBS)
Viterbi.o: Viterbi.cc json.hpp
	g++ -c Viterbi.cc -std=c++17
clean :
	rm *.o Viterbi
