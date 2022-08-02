#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm> // for reverse vector

size_t nCodons;

std::vector<std::string> splitstr(std::string str, std::string deli = " ") {
    int start = 0;
    int end = str.find(deli);
    std::vector<std::string> result;
    while (end != -1) {
        auto item = str.substr(start, end - start);
        result.push_back(item);
        start = end + deli.size();
        end = str.find(deli, start);
    }
    auto item = str.substr(start, end - start);
    result.push_back(item);
    return (result);
}
std::string state_id_to_description(size_t id) {
    std::vector<std::string> states {"ig5", "stA", "stT", "stG"};
    for (size_t i = 0; i < nCodons; i++) {
        for (size_t j = 0; j < 3; j++) {
            states.push_back("c" + std::to_string(i) + "," + std::to_string(j));
        }
    }
    states.push_back("st1");
    states.push_back("st2");
    states.push_back("st3");
    states.push_back("ig3");
    for (size_t i = 0; i < nCodons + 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            states.push_back("i" + std::to_string(i) +","+ std::to_string(j));
        }
    }
    states.push_back("ter1");
    states.push_back("ter2");
    return states[id];
}

std::vector<float> read_I(std::string path) {
    std::ifstream myfile (path);
    std::string line;
    std::vector<float> I;

    if(!myfile.is_open()) {
        std::cout << "error opening " << path << '\n';
        exit(EXIT_FAILURE);
    }
    while(getline(myfile, line)) {
        I.push_back(std::stof(splitstr(line, ";")[1]));
    }
    myfile.close();
    return(I);
}

std::vector<std::vector<float>> read_A(std::string path) {
    std::ifstream myfile (path);
    std::string line;
    std::vector<std::vector<float>> A;

    if(!myfile.is_open()) {
        std::cout << "error opening " << path << '\n';
        exit(EXIT_FAILURE);
    }
    int last_row = 0;
    std::vector<float> row;
    while(getline(myfile, line)) {
        auto not_tidy_row = splitstr(line, ",")[0];
        auto current_row = std::stoi(not_tidy_row.substr(1,not_tidy_row.size()));
        if (last_row != current_row) {
            A.push_back(row);
            last_row = current_row;
            row.clear();
        }
        row.push_back(std::stof(splitstr(line, ";")[1]));
    }
    A.push_back(row);
    return A;
}
std::vector<std::vector<std::vector<std::vector<float>>>> read_B(std::string path) {
    std::ifstream myfile (path);
    std::string line;
    std::vector<std::vector<std::vector<std::vector<float>>>> B;

    if(!myfile.is_open()) {
        std::cout << "error opening " << path << '\n';
        exit(EXIT_FAILURE);
    }

    int last_level_2 = 0; // third index
    int last_level_1 = 0;
    int last_level_0 = 0; // first index
    std::vector<std::vector<std::vector<float>>> level_1_row;
    std::vector<std::vector<float>> level_2_row;
    std::vector<float> level_3_row;
    while(getline(myfile, line)) {
        auto current_level_2 = std::stoi(splitstr(line, ",")[2].substr(1,1));
        auto current_level_1 = std::stoi(splitstr(line, ",")[1].substr(1,1));
        auto not_tidy_level_0 = splitstr(line, ",")[0];
        auto current_level_0 = std::stoi(not_tidy_level_0.substr(1,not_tidy_level_0.size()));
        if (last_level_2 != current_level_2) {
            level_2_row.push_back(level_3_row);
            last_level_2 = current_level_2;
            level_3_row.clear();
        }
        if (last_level_1 != current_level_1) {
            level_1_row.push_back(level_2_row);
            last_level_1 = current_level_1;
            level_2_row.clear();
        }
        if (last_level_0 != current_level_0) {
            B.push_back(level_1_row);
            last_level_0 = current_level_0;
            level_1_row.clear();
        }
        level_3_row.push_back(std::stof(splitstr(line, ";")[1]));
    }
    level_2_row.push_back(level_3_row);
    level_1_row.push_back(level_2_row);
    B.push_back(level_1_row);
    return B;
}
std::vector<std::vector<int>> read_seqs(std::string path) {
    std::ifstream myfile (path);
    std::string line;
    std::vector<std::vector<int>> seqs;

    if(!myfile.is_open()) {
        std::cout << "error opening " << path << '\n';
        exit(EXIT_FAILURE);
    }

    std::vector<int> seq;
    while(getline(myfile, line)) {
        // todo: does this cause an error if line is empty, (it should always contain a newline, right?)
        if (line[0] == '>') {
            seqs.push_back(seq);
            seq.clear();
        }
        for (auto c : line) {
            switch (c) {
                case 'A':
                    seq.push_back(0);
                    break;
                case 'C':
                    seq.push_back(1);
                    break;
                case 'G':
                    seq.push_back(2);
                    break;
                case 'T':
                    seq.push_back(3);
                    break;
            }
        }
    }
    seqs.push_back(seq);
    seqs.erase(seqs.begin());
    return seqs;
}

void print(std::vector<float> v) {
    for (auto c : v) {
        std::cout << c << '\n';
    }
}
void print(std::vector<std::vector<float>> a) {
    std::cout << "-\t";
    for (size_t j = 0; j < a.size(); ++j) {
        std::cout << state_id_to_description(j) << '\t';
    }
    std::cout << '\n';
    for (size_t i = 0; i < a.size(); i++) {
        std::cout << state_id_to_description(i) << '\t';
        for (size_t j = 0; j < a[i].size(); j++) {
            if (i == j) std::cout << "\033[92m"<< std::round(a[i][j]*100)/100 << "\033[0m" << "\t";
            // else if (state_id_to_description(j).at(0) == 'i' && a[i][j] != 0)  std::cout << "\033[93m"<< std::round(a[i][j]*100)/100 << "\033[0m" << "\t";
            else std::cout << std::round(a[i][j]*100)/100 << "\t";
        }
        std::cout << '\n';
    }
}
void print(std::vector<std::vector<std::vector<std::vector<float >>>> b) {
    for (size_t i = 0; i < b.size(); i++) {
        for (size_t j = 0; j < b[i].size(); j++) {
            std::cout << "i,j = " << i << "," << j << '\n';
            for (auto row : b[i][j]) {
                for (auto v : row) {
                    std::cout << std::round(v*100)/100 << '\t';
                }
                std::cout << '\n';
            }
        }
    }
}
void print(std::vector<int> seq, std::string sep) {
    for (auto base : seq) {
        char b;
        switch (base) {
            case 0:
                b = 'A';
                break;
            case 1:
                b = 'C';
                break;
            case 2:
                b = 'G';
                break;
            case 3:
                b = 'T';
                break;
        }
        std::cout << b << sep;
    }
    std::cout << '\n';
}
void print(std::vector<std::vector<int>> seqs, std::string sep) {
    for (auto seq : seqs) {
        print(seq, sep);
    }
}
void print(std::vector<size_t> state_seq) {
    for (auto state: state_seq) {
        std::cout << state_id_to_description(state) << "\t";
    }
    std::cout << '\n';
}

std::vector<size_t> viterbi(std::vector<std::vector<float>> A,
                         std::vector<std::vector<std::vector<std::vector<float>>>> B,
                         std::vector<float> I,
                         std::vector<int> Y)
{
    auto nStates = A.size();
    auto n = Y.size();
    auto order = 2;
    std::vector<int> y_old {4,4};
    // i, state
    std::vector<std::vector<float>> dp_g;
    std::vector<std::vector<size_t>> dp_g_pointer_to_max;

    std::vector<float> icolumn;
    std::vector<size_t> icolumn_max(nStates);
    for (size_t state = 0; state < nStates; state++) {
        auto value = std::log(I[state] * B[state][y_old[0]][y_old[1]][Y[0]]);
        icolumn.push_back(value);
    }
    dp_g.push_back(icolumn);
    icolumn.clear();
    // dp_g_pointer_to_max.push_back(icolumn_max);
    // icolumn_max.clear();
    for (size_t i = 1; i < n; i++) {
        y_old.erase(y_old.begin());
        y_old.push_back(Y[i-1]);// das war fälschilicher weise in der for schleufe,, ripriprip
        for (size_t q = 0; q < nStates; q++) {
            auto M = std::log(0);
            size_t max_state = 0;
            for (size_t state = 0; state < nStates; state++) {
                // size_t my_state = 2;
                // size_t my_prev_state = 1;
                // size_t my_i = 2;
                // if (i == my_i && q == my_state && state == my_prev_state) {
                //     std::cout << "A = " << A[my_prev_state][my_state] << '\n';
                //     std::cout << "dp_g = " << dp_g[i-1][my_prev_state] << '\n';
                //     auto mm = std::log(A[state][q]) + dp_g[i-1][state];
                //     std::cout << "m = " << mm << '\n';
                //     std::cout << "b index = " << " " << q << " " << y_old[0]<< " " << y_old[1]<< " " << Y[i]<< '\n';
                //     std::cout << "b = " << B[q][y_old[0]][y_old[1]][Y[i]] << '\n';
                // }
                auto m = std::log(A[state][q]) + dp_g[i-1][state];
                if (m > M) {
                    M = m;
                    max_state = state;
                }
            }
            icolumn.push_back(std::log(B[q][y_old[0]][y_old[1]][Y[i]]) + M);
            // icolumn_max.push_back(max_state);
        }
        dp_g.push_back(icolumn);
        icolumn.clear();
        // dp_g_pointer_to_max.push_back(icolumn_max);
        // icolumn_max.clear();
    }


    // for (size_t i = 0; i < n; i++) {
    //     std::cout << "i = " << i << '\t';
    //     for (auto value : dp_g[i]) {
    //         std::cout << std::round(value*100)/100 << '\t';
    //     }
    //     std::cout << '\n';
    // }

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
            auto m = dp_g[i][state] + std::log(A[state][x[x.size()-1]]);
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
float log_prob_of_state_seq(std::vector<std::vector<float>> A,
                         std::vector<std::vector<std::vector<std::vector<float>>>> B,
                         std::vector<float> I,
                         std::vector<size_t> state_seq,
                         std::vector<int> Y)
{
    if (Y.size() != state_seq.size()) {
        std::cout << "state_seq and Y have differing lenghts" << '\n';
        return std::log(0);
    }
    auto n = Y.size();
    std::vector<int> y_old {4,4};

    float prob = std::log(I[state_seq[0]] * B[state_seq[0]][y_old[0]][y_old[1]][Y[0]]);

    for (size_t i = 1; i < n; i++) {
        y_old.erase(y_old.begin());
        y_old.push_back(Y[i-1]);
        prob += std::log(A[state_seq[i-1]][state_seq[i]] * B[state_seq[i]][y_old[0]][y_old[1]][Y[i]]);
    }
    return prob;
}
void write_to_file(std::vector<std::vector<int>> seqs, std::vector<std::vector<size_t>> state_seqs) {
    std::ofstream file;
    file.open("viterbi." + std::to_string(nCodons) + "codons.csv");
    for (size_t i = 0; i < seqs.size(); i++) {
        auto seq = seqs[i];
        auto state_seq = state_seqs[i];
        if (seq.size() != state_seq.size()) {
            std::cout << "seq and state_seq at " << i << " have differing lenghts" << '\n';
            exit(EXIT_FAILURE);
        }
        for (auto base : seq) {
            char b;
            switch (base) {
                case 0:
                    b = 'A';
                    break;
                case 1:
                    b = 'C';
                    break;
                case 2:
                    b = 'G';
                    break;
                case 3:
                    b = 'T';
                    break;
            }
            file << b << "\t";
        }
        file << "\n";

        for (auto state : state_seq) {
            file << state_id_to_description(state) << "\t";
        }
        file << "\n";
        file << "-\n";
    }
    file.close();
}

void write_to_file_atg_aligned(std::vector<std::vector<int>> seqs, std::vector<std::vector<size_t>> state_seqs) {
    std::ofstream file;
    file.open("viterbi." + std::to_string(nCodons) + "codons.atg_aligned.csv");

    size_t max_index_of_start_a = 0;
    for (size_t i = 0; i < seqs.size(); i++) {
        for (size_t j = 0; j < state_seqs[i].size(); j++) {
            if (state_seqs[i][j] == 1) {
                if (j > max_index_of_start_a) {
                    max_index_of_start_a = j;
                }
                break;
            }
        }
    }
    for (size_t i = 0; i < seqs.size(); i++) {
        auto seq = seqs[i];
        auto state_seq = state_seqs[i];

        if (seq.size() != state_seq.size()) {
            std::cout << "seq and state_seq at " << i << " have differing lenghts" << '\n';
            exit(EXIT_FAILURE);
        }
        size_t index_of_start_a = 0;
        for (size_t j = 0; j < seq.size(); j++) {
            if (state_seq[j] == 1) {
                index_of_start_a = j;
                break;
            }
        }
        file << "seq " << i << "\n";

        for (size_t j = 0; j < max_index_of_start_a - index_of_start_a; ++j) {
            file << "-\t";
        }

        for (auto base : seq) {
            char b;
            switch (base) {
                case 0:
                    b = 'A';
                    break;
                case 1:
                    b = 'C';
                    break;
                case 2:
                    b = 'G';
                    break;
                case 3:
                    b = 'T';
                    break;
            }
            file << b << "\t";
        }
        file << "\n";

        for (size_t j = 0; j < max_index_of_start_a - index_of_start_a; ++j) {
            file << "-\t";
        }

        for (auto state : state_seq) {
            file << state_id_to_description(state) << "\t";
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char *argv[]) {
    char * path;
    if (argc != 3) {
        std::cout << "usage: ./main path/to/fasta/file nCodons_used_in_model" << '\n';
        return(1);
    }
    else {
        path = argv[1];
        std::string str_nCodons(argv[2]);
        nCodons = std::stoi(str_nCodons);
    }

    auto I = read_I("I."+std::to_string(nCodons)+"codons.txt");
    // print(I);
    auto A = read_A("A."+std::to_string(nCodons)+"codons.txt");
    // print(A);
    auto B = read_B("B."+std::to_string(nCodons)+"codons.txt");
    // print(B);
    auto seqs = read_seqs(path);
    // print(seqs);

    std::vector<std::vector<size_t>> state_seqs;
    for (auto seq : seqs) {
        // print(seq, "\t");
        auto x = viterbi(A,B,I,seq);
        // print(x);
        state_seqs.push_back(x);
        // std::cout << "-" << '\n';
    }
    write_to_file_atg_aligned(seqs, state_seqs);
    write_to_file(seqs, state_seqs);
}
