#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm> // for reverse vector
#include "json.hpp"
using json = nlohmann::json;
size_t nCodons;

std::vector<size_t> viterbi(json A, json B, json I, json Y) {
    auto nStates = A.size();
    auto n = Y.size();
    // i, state
    std::vector<std::vector<float>> dp_g;
    std::vector<std::vector<size_t>> dp_g_pointer_to_max;

    std::vector<float> icolumn;
    std::vector<size_t> icolumn_max(nStates);
    for (size_t state = 0; state < nStates; state++) {
        auto value = std::log(I[0][state].get<float>() * B[Y[0].get<int>()][state].get<float>());
        icolumn.push_back(value);
    }
    dp_g.push_back(icolumn);
    icolumn.clear();
    // dp_g_pointer_to_max.push_back(icolumn_max);
    // icolumn_max.clear();
    for (size_t i = 1; i < n; i++) {
        for (size_t q = 0; q < nStates; q++) {
            auto M = std::log(0);
            size_t max_state = 0;
            for (size_t state = 0; state < nStates; state++) {
                auto m = std::log(A[state][q].get<float>()) + dp_g[i-1][state];
                if (m > M) {
                    M = m;
                    max_state = state;
                }
            }
            icolumn.push_back(std::log(B[Y[i].get<int>()][q].get<float>()) + M);
            // icolumn_max.push_back(max_state);
        }
        dp_g.push_back(icolumn);
        icolumn.clear();
        // dp_g_pointer_to_max.push_back(icolumn_max);
        // icolumn_max.clear();
    }

    // backtracking
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

    int n_as_int = static_cast<int>(n);
    for (int i = n_as_int-2; i != -1; i--) {
        int max_state = 0;
        float max_value = std::log(0);
        for (size_t state = 0; state < nStates; state++) {
            auto m = dp_g[i][state] + std::log(A[state][x[x.size()-1]].get<float>());
            if (m > max_value) {
                max_value = m;
                max_state = state;
            }
        }
        x.push_back(max_state);
    }
    std::reverse(x.begin(), x.end());
    return x;
}

void is_empty(std::ifstream& pFile, std::string & s) {
    if (!pFile) {
        std::cout << s << " is not open" << '\n';
    }
    if (pFile.peek() == std::ifstream::traits_type::eof()) {
        std::cout << s << " is empty" << '\n';
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "you need to pass nCodons" << '\n';
        return 1;
    }
    std::string str_nCodons(argv[1]);
    nCodons = std::stoi(str_nCodons);

    std::vector<std::vector<size_t>> state_seqs;

    ////////////////////////////////////////////////////////////////////////////////
    // [nSeqs, lengths], values in range(rows in B)
    std::string seq_path = "output/" + std::to_string(nCodons) + "codons/out.seqs." + std::to_string(nCodons) + "codons.fa.json";
    std::ifstream f_seq(seq_path);
    is_empty(f_seq, seq_path);

    json seqs;
    f_seq >> seqs;
    // std::cout << "seqs = " << seqs.dump() << '\n';

    ////////////////////////////////////////////////////////////////////////////////
    // [1, nStates]
    std::string i_path = "output/" + std::to_string(nCodons) + "codons/I." + std::to_string(nCodons) + "codons.csv.json";
    std::ifstream f_i(i_path);
    is_empty(f_i, i_path);

    json i;
    f_i >> i;
    // std::cout << "i = " << i.dump() << '\n';
    // std::vector<float> i_v(i[0]);
    ////////////////////////////////////////////////////////////////////////////////
    // [nStates, nStates]
    std::string a_path = "output/" + std::to_string(nCodons) + "codons/A." + std::to_string(nCodons) + "codons.csv.json";
    std::ifstream f_a(a_path);
    is_empty(f_a, a_path);

    json a;
    f_a >> a;
    // std::cout << "a = " << a.dump() << '\n';
    ////////////////////////////////////////////////////////////////////////////////
    // [emissions_state_size, nStates]
    std::string b_path = "output/" + std::to_string(nCodons) + "codons/B." + std::to_string(nCodons) + "codons.csv.json";
    std::ifstream f_b(b_path);
    is_empty(f_b, b_path);

    json b;
    f_b >> b;
    // std::cout << "b = " << b.dump() << '\n';
    ////////////////////////////////////////////////////////////////////////////////

    for (auto seq : seqs) {
        auto x = viterbi(a,b,i,seq);
        state_seqs.push_back(x);
    }

    std::string out_path = "output/" + std::to_string(nCodons) + "codons/viterbi.json";
    std::ofstream f_out(out_path);
    f_out << json(state_seqs);
    f_out.close();

}
