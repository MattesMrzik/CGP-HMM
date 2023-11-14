#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <memory>
#include <cmath>
#include <algorithm> // for reverse vector
#include "json.hpp"
#include <boost/program_options.hpp>
#include <future>
#include <thread>
#include <chrono>
using json = nlohmann::json;


std::tuple<float, float> get_M_and_emission_prob_for_q(const std::vector<std::vector<float>> & A,
                                    const std::vector<std::vector<float>> & B,
                                    const std::vector<std::vector<float>> & Y,
                                    const std::vector<std::vector<float>> & dp_g,
                                    size_t i,
                                    size_t q) {
    auto M = std::log(0);
    for (size_t state = 0; state < A.size(); state++) {
        auto m = std::log(A[state][q]) + dp_g[i-1][state];
        if (m > M) {
            M = m;
        }
    }
    float emission_prob = 0;
    for (size_t emission_id = 0; emission_id < B.size(); emission_id++) {
        if (Y[i][emission_id] > 0) {
            emission_prob += B[emission_id][q] * Y[i][emission_id];
        }

    }

    return std::tuple<float, float>{M, emission_prob};
 }


// not shared ptr but ref to const vector
std::vector<size_t> viterbi(const std::vector<float> & I,
                            const std::vector<std::vector<float>> & A,
                            const std::vector<std::vector<float>> & B,
                            const std::vector<std::vector<float>> & Y,
                            int id, size_t nThreads, bool only_first, bool verbose,
                            std::string out_path) {

    // TODO maybe also pass the results vector x by ref
    size_t nStates = A.size();

    if (nStates < nThreads) {
        std::cout << "there are fewer states than threads" << '\n';
        nThreads = nStates;
    }
    size_t n = Y.size();
    size_t emission_size = B.size();


    std::ofstream eta_file;
    if (only_first) {
        std::filesystem::path new_path = std::filesystem::path(out_path).remove_filename() / "viterbi_eta.txt";
        std::cout << "eta path " << new_path << "\n";
        eta_file.open(new_path);
        eta_file << "start" << "\n";
    }

    // i, state
    std::vector<std::vector<float>> dp_g;
    std::vector<float> icolumn;
    std::vector<size_t> icolumn_max(nStates);
    // for different states this could be split across multiple jobs

    // O(|Q| * |E|) , normal viterbi doenst have E, but i have multi hot
    for (size_t state = 0; state < nStates; state++) {
        float emission_prob = 0;
        for (size_t emission_id = 0; emission_id < emission_size; emission_id++) {
            if (Y[0][emission_id] > 0) {
                emission_prob += I[state] * B[emission_id][state] * Y[0][emission_id];
            }
        }
        emission_prob = std::log(emission_prob);
        icolumn.push_back(emission_prob);
    }
    dp_g.push_back(icolumn);
    icolumn.clear();

    if (only_first && verbose) {
        eta_file << "init column done" << "\n";
    }

    int bar_size = 50;
    // O (n * |Q| * (|Q| + |E|)/nThreads)
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < n; i++) {
        if ((nThreads != 1 || only_first) && verbose) { // bc if it one, then seqs might get computed in parallel, which would mess with the print
            std::string done(i*bar_size/n, '#');
            std::string not_done(bar_size - i*bar_size/n, ' ');
            std::cout << "[" << done << not_done << "]" << '\r';
        }
        for (size_t q = 0; q < nStates;   ) {
            float M, emission_prob;
            if (nThreads == 1) {
                std::tie(M, emission_prob) = get_M_and_emission_prob_for_q(A,B,Y, dp_g ,i, q);
                icolumn.push_back(std::log(emission_prob) + M);
                q++;
            }
            else {
                std::vector<std::future<std::tuple<float,float>>> threads;
                for (size_t thread_id = 0; thread_id < nThreads; thread_id++) {
                    threads.push_back(std::async(std::launch::async, get_M_and_emission_prob_for_q, A,B,Y, dp_g, i, q + thread_id));
                }
                for (size_t thread_id = 0; thread_id < nThreads; thread_id++, q++) {
                    std::tie(M, emission_prob) = threads[thread_id].get();
                    icolumn.push_back(std::log(emission_prob) + M);
                }
            }
        }
        dp_g.push_back(icolumn);
        icolumn.clear();

        if (only_first && i % 100 == 0) {
            auto stop = std::chrono::high_resolution_clock::now();
            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            auto eta = duration.count() * 1.0/ i * (n - i);
            auto eta_seconds = eta * 1.0/1000000000;
            auto eta_minutes = eta_seconds/60;

            eta_file << i<< "/" << n << ",\teta in seconds =  " << eta * 1.0/1000000000 << ",\teta in minutes = " << eta_minutes << "\n";
        }
    }

    // backtracking

    // O(|Q|)
    std::vector<size_t> x;
    size_t last_state = 0;
    float max_prob_last_state = std::log(0);
    for (size_t state = 0; state < nStates; state++) {
        if (dp_g[n-1][state] > max_prob_last_state) {
            last_state = state;
            max_prob_last_state = dp_g[n-1][state];
        }
    }
    x.push_back(last_state);

    // O(n * |Q|)
    int n_as_int = static_cast<int>(n);
    for (int i = n_as_int-2; i != -1; i--) {
        int max_state = 0;
        float max_value = std::log(0);
        for (size_t state = 0; state < nStates; state++) {
            auto m = dp_g[i][state] + std::log(A[state][x[x.size()-1]]);
            if (m > max_value) {
                max_value = m;
                max_state = state;
            }
        }
        x.push_back(max_state);
    }
    std::reverse(x.begin(), x.end());
    if (verbose) {
        std::string all_done(bar_size, '#');
        std::cout << "[" << all_done << "]" << '\r';
        std::cout << '\n';
    }
    return x;
}

void is_empty(std::ifstream& pFile, std::string & s) {
    if (!pFile) {
        std::cerr << s << " is not open" << '\n';
        std::cerr <<"\n\n\n\n\n\n\n\n\n\n\n";
        exit (1);
    }
    if (pFile.peek() == std::ifstream::traits_type::eof()) {
        std::cerr << s << " is empty" << '\n';
        std::cerr <<"\n\n\n\n\n\n\n\n\n\n\n";
        exit (1);
    }
}

int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("seqs_path", po::value<std::string>(), "input seqs file path")
        ("dir_path_for_para", po::value<std::string>(), "path to dir which contains I.json, A.json, B.json")
        ("out_path", po::value<std::string>(), "out file name")
        ("parallel_seqs", "calculate seqs in parallel instead of Ms")
        ("only_first_seq", "calculate only the first seq")
        ("verbose", "verbose")
        ("nThreads,j", po::value<int>(), "n threads");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

////////////////////////////////////////////////////////////////////////////////
    int nThreads;
    if (vm.count("nThreads")) {
        nThreads = vm["nThreads"].as<int>();
    }
    else {
        std::cout << "Using only 1 thread. Consider -j" << '\n';
    }
////////////////////////////////////////////////////////////////////////////////
    bool only_first_seq = vm.count("only_first_seq");

    bool verbose = vm.count("verbose");
////////////////////////////////////////////////////////////////////////////////
    std::string seqs_path;
    if (vm.count("seqs_path")) seqs_path = vm["seqs_path"].as<std::string>();
    else {
        std::cout << "You must specify the seqs_path\n";
        return 1;
    }
////////////////////////////////////////////////////////////////////////////////
    std::string dir_path_for_para;
    if (vm.count("dir_path_for_para")) dir_path_for_para = vm["dir_path_for_para"].as<std::string>();
    else {
        std::cout << "You must specify the dir_path_for_para\n";
        return 1;
    }

    auto i_path = dir_path_for_para + "/I.json";
    auto a_path = dir_path_for_para + "/A.json";
    auto b_path = dir_path_for_para + "/B.json";
////////////////////////////////////////////////////////////////////////////////
    std::string out_path;
    if (vm.count("out_path")) out_path = vm["out_path"].as<std::string>();
    else {
        std::cout << "You must specify the out_path\n";
        return 1;
    }

    // [nSeqs, seq_len, emissions_alphabet_size]
    std::ifstream f_seq(seqs_path);
    is_empty(f_seq, seqs_path);
    json seqs;
    f_seq >> seqs;
    std::vector<std::vector<std::vector<float>>> seqs_v = seqs.get<std::vector<std::vector<std::vector<float>>>>();
    ////////////////////////////////////////////////////////////////////////////
    // [1, nStates]
    std::ifstream f_i(i_path);
    is_empty(f_i, i_path);
    json i;
    f_i >> i;
    std::vector<float> i_v = i.get<std::vector<std::vector<float>>>()[0];
    // std::cout << "i = " << i.dump() << '\n';
    ////////////////////////////////////////////////////////////////////////////
    // [nStates, nStates]
    std::ifstream f_a(a_path);
    is_empty(f_a, a_path);
    json a;
    f_a >> a;
    std::vector<std::vector<float>> a_v = a.get<std::vector<std::vector<float>>>();
    // std::cout << "a = " << a.dump() << '\n';
    ////////////////////////////////////////////////////////////////////////////
    // [emissions_state_size, nStates]
    std::ifstream f_b(b_path);
    is_empty(f_b, b_path);
    json b;
    f_b >> b;
    std::vector<std::vector<float>> b_v = b.get<std::vector<std::vector<float>>>();
    // std::cout << "b = " << b.dump() << '\n';
    ////////////////////////////////////////////////////////////////////////////

    std::vector<std::vector<size_t>> state_seqs;
    for (size_t j = 0; j < seqs.size(); ++j) {
        state_seqs.push_back({});
    }

    if (vm.count("parallel_seqs")) {
        if (only_first_seq) {
            std::cout << "discading option --only_first_seq since you passed --parallel_seqs" << '\n';
        }
        for (size_t low = 0; low < seqs.size(); low += nThreads) {
            std::vector<std::future<std::vector<size_t> > > threads;
            size_t now_using = 0;
            if (low + nThreads <= seqs.size()) {
                now_using = nThreads;
            }
            else {
                now_using = seqs.size() - low;
            }
            for (size_t j = 0; j < now_using; j++) {
                threads.push_back(std::async(std::launch::async, viterbi, i_v,a_v,b_v, seqs_v[low + j], low + j, 1, only_first_seq, verbose, out_path));
            }
            for (size_t j = 0; j < now_using; j++) {
                auto result = threads[j].get();
                state_seqs[low + j] = result;
            }
        }
    }
    else {
        for (size_t seq_id = 0; seq_id < seqs.size(); seq_id++) {
            if (only_first_seq && seq_id > 0) {
                std::cout << "Calculated only first sequence. Check whether the result has the same len as desired fasta len." << '\n';
                break;
            }
            state_seqs[seq_id] = viterbi(i_v, a_v, b_v, seqs_v[seq_id], seq_id, nThreads, only_first_seq, verbose, out_path);
            if (verbose){
                std::cout << "Done with seq " << seq_id << '\n';
            }
        }
    }

    std::ofstream f_out(out_path);
    if (only_first_seq) {
        std::vector<std::vector<size_t>> first_seq_wrapped;
        first_seq_wrapped.push_back(state_seqs[0]);
        f_out << json(first_seq_wrapped);
    }
    else {
        f_out << json(state_seqs);
    }
    f_out.close();
}
