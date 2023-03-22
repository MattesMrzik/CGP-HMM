#!/usr/bin/env python3
import os
import pandas as pd
import json
import re
import time

def run_cc_viterbi(config):
    import multiprocessing
    start = time.perf_counter()
    seq_path = f"--seq_path {config.fasta_path}.json" if config.manual_passed_fasta else ""
    only_first_seq = f"--only_first_seq" if config.only_first_seq else ""
    command = f"{config.src_path}/Viterbi -c {config.nCodons} -j {multiprocessing.cpu_count()-1} {seq_path} {only_first_seq}"
    print("starting", command)
    os.system(command)
    print("done viterbi. it took ", time.perf_counter() - start)

def get_true_state_seqs_from_true_MSA(config):
    # calc true state seq from true MSA
    true_state_seqs = []
    msa_state_seq = ""
    with open(f"{config.src_path}/output/{config.nCodons}codons/trueMSA.txt","r") as msa:
        for j, line in enumerate(msa):
            true_state_seq = []
            if j == 0:
                msa_state_seq = line.strip()
                continue
            which_codon = 0
            start_foud = False
            stop_foud = False
            i = 0
            while i < len(line.strip()):
                if line[i] in ["-", "i"]:
                    i += 1
                    continue
                if line[i] == "d":
                    which_codon += 1
                    i += 3
                    continue
                if msa_state_seq[i] == "A":
                    start_id = config.model.str_to_state_id("stA")
                    true_state_seq += [start_id, start_id + 1, start_id + 2]
                    start_foud = True
                    i += 3
                elif msa_state_seq[i] == "S":
                    stop_id = config.model.str_to_state_id("stop1")
                    true_state_seq += [stop_id, stop_id + 1, stop_id + 2]
                    stop_foud = True
                    i += 3
                elif msa_state_seq[i] == "i":
                    insert_id = config.model.str_to_state_id(f"i_{which_codon},0")
                    true_state_seq += [insert_id, insert_id + 1, insert_id + 2]
                    i += 3
                else:
                    if stop_foud:
                        true_state_seq += [config.model.str_to_state_id("ig3'")]
                        i += 1
                    elif not start_foud:
                        true_state_seq += [config.model.str_to_state_id("ig5'")]
                        i += 1
                    else:
                        codon_id = config.model.str_to_state_id(f"c_{which_codon},0")
                        true_state_seq += [codon_id, codon_id + 1, codon_id + 2]
                        which_codon += 1
                        i += 3
            if len(true_state_seqs) == 0:
                true_state_seqs = [true_state_seq]
            else:
                true_state_seqs.append(true_state_seq)
    return true_state_seqs

def load_viterbi_guess(config):
    viterbi_file = open(f"{config.src_path}/output/{config.nCodons}codons/viterbi_cc_output.json", "r")
    viterbi = json.load(viterbi_file)
    return viterbi

def write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi):
    def state_seq_to_nice_str(state_seq): #[0,0,0,0,0,1,2,3,4,5,6,7,8,9,16,16,16,16]
        s = ""
        for i, state in enumerate(state_seq):
            if state == config.model.str_to_state_id("ig5'"):
                s += "5"
            if state == config.model.str_to_state_id("stA"):
                s += "A"
            if state == config.model.str_to_state_id("stT"):
                s += "T"
            if state == config.model.str_to_state_id("stG"):
                s += "G"
            if state == config.model.str_to_state_id("stop1"):
                s += "S"
            if state == config.model.str_to_state_id("stop2"):
                s += "T"
            if state == config.model.str_to_state_id("stop3"):
                s += "P"
            if state == config.model.str_to_state_id("ig3'"):
                s += "3"
            # insert or codon
            if config.model.state_id_to_str(state)[-1] == "0": # and len(config.model.state_id_to_str(state)) == 5
                splitted = re.split("[_|,]",config.model.state_id_to_str(state))
                s += splitted[0] + (splitted[1] + (" " if len(splitted[1]) < 2 else ""))[:2]
        return s

    # state_str      555ATGc0 STP33
    # msa_seq_str ---ATCATGCTATAGTA-----
    def expand_nice_states_str_to_fit_msa(state_str, msa_seq_str):
        id_in_state_str = 0
        new_str = ""
        for cc in msa_seq_str:
            if cc == "-":
                new_str += "-"
            elif cc == "\n":
                new_str += "\n"
            elif cc in ["i", "d"]:
                new_str += " "
            else:
                new_str += state_str[id_in_state_str]
                id_in_state_str += 1
        return new_str

    with open(f"{config.src_path}/output/{config.nCodons}codons/trueMSA.txt","r") as msa:
        with open(f"{config.src_path}/output/{config.nCodons}codons/trueMSA_viterbi.txt","w") as out:
            for i, msa_seq_str in enumerate(msa):
                seq_id = i - 1
                if seq_id < 0:
                    out.write("# first line is msa state seq. the first of the 3 tuple is aligned seq, second is true state seq, third is viterbi guess\n")
                    out.write(msa_seq_str)
                    continue
                else:
                    out.write(msa_seq_str) # ----TTATGTTCTAATCGGTT from useMSAgen trueMSA
                    add_one_terminal_symbol = True
                    one = 1 if add_one_terminal_symbol else 0
                    try:
                        x = viterbi[seq_id]
                    except:
                        print("for this seq_id there is no viterbi guess")
                        break
                    assert len(viterbi[seq_id]) - one == len(true_state_seqs[seq_id]), f"length if viterbi {viterbi[seq_id]} and true state seq {true_state_seqs[seq_id]} differ, check if parallel computing works as intended"
                    out.write(expand_nice_states_str_to_fit_msa(state_seq_to_nice_str(true_state_seqs[seq_id]), msa_seq_str))
                    out.write(expand_nice_states_str_to_fit_msa(state_seq_to_nice_str(viterbi[seq_id]), msa_seq_str))
                out.write("\n")

def eval_start_stop(config, viterbi):
    start = time.perf_counter()
    print("starting eval viterbi")

    stats = {"start_not_found" : 0,\
             "start_too_early" : 0,\
             "start_correct" : 0,\
             "start_too_late" : 0,\
             "stop_not_found" : 0,\
             "stop_too_early" : 0,\
             "stop_correct" : 0,\
             "stop_too_late" : 0}

    start_stop = pd.read_csv(f"{config.src_path}/output/{config.nCodons}codons/start_stop_pos.txt", sep=";", header=None)
    stA_id = config.model.str_to_state_id("stA")
    stop1_id = config.model.str_to_state_id("stop1")
    for i, state_seq in enumerate(viterbi):
        try:
            viterbi_start = state_seq.index(stA_id)
        except:
            viterbi_start = -1
        try:
            viterbi_stop = state_seq.index(stop1_id)
        except:
            viterbi_stop = -1


        true_start = start_stop.iloc[i,1]
        true_stop = start_stop.iloc[i,2]

        # print(f"true_start = {true_start} vs viterbi_start = {viterbi_start}")
        # print(f"true_stop = {true_stop} vs viterbi_stop = {viterbi_stop}")

        nSeqs = len(viterbi)

        if viterbi_start == -1:
            stats["start_not_found"] += 1/nSeqs
            if viterbi_stop != -1:
                print("found stop but not start")
                quit(1)
        elif viterbi_start < true_start:
            stats["start_too_early"] += 1/nSeqs
        elif viterbi_start == true_start:
            stats["start_correct"] += 1/nSeqs
        else:
            stats["start_too_late"] += 1/nSeqs

        if viterbi_stop == -1:
            stats["stop_not_found"] += 1/nSeqs
        elif viterbi_stop < true_stop:
            stats["stop_too_early"] += 1/nSeqs
        elif viterbi_stop == true_stop:
            stats["stop_correct"] += 1/nSeqs
        else:
            stats["stop_too_late"] += 1/nSeqs


    with open(f"{config.src_path}/output/{config.nCodons}codons/statistics.json", "w") as file:
        json.dump(stats, file)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("done evaluating viterbi. it took ", time.perf_counter() - start)

def compare_guess_to_true_state_seq(trues, guesses):
    correct = 0
    false = 0
    for true, guess in zip(trues, guesses):
        for i, (x,y) in enumerate(zip(true, guess)):
            if x != y:
                false += 1
                print("true seq")
                print(true[:i], "__first_mistake__", true[i:])
                print("prediction")
                print(guess[:i], "__first_mistake__", guess[i:])
                break
        else:
            correct += 1
    print(f"correct = {correct}, false = {false}, accuracy = {correct/(false+correct)}")

if __name__ == "__main__":
    from Config import Config
    import numpy as np
    config = Config("main_programm_dont_interfere")

    dir_path = f"{config.src_path}/output/{config.nCodons}codons/after_fit_matrices"

    I_path =f"{dir_path}/I.json"
    A_path =f"{dir_path}/A.json"
    B_path =f"{dir_path}/B.json"

    if not os.path.exists(A_out_path):
        # from cell.py
        def read_weights_from_file(kernel_dir):
            with open(f"{kernel_dir}/I_kernel.json") as file:
                I_kernel = np.array(json.load(file))
            with open(f"{kernel_dir}/A_kernel.json") as file:
                A_kernel = np.array(json.load(file))
            with open(f"{kernel_dir}/B_kernel.json") as file:
                B_kernel = np.array(json.load(file))
            return I_kernel, A_kernel, B_kernel

        kernel_dir = f"{config.src_kernel_dir}/output/{config.nCodons}codons/after_fit_kernels/"
        I_kernel, A_kernel, B_kernel = read_weights_from_file(kernel_dir)

        config.model.I_as_dense_to_json_file(f"{dir_path}/I.json", I_kernel)
        config.model.A_as_dense_to_json_file(f"{dir_path}/A.json", A_kernel)
        config.model.B_as_dense_to_json_file(f"{dir_path}/B.json", B_kernel)

    if os.path.exists(f"{config.src_path}/output/{config.nCodons}codons/viterbi_cc_output.json"):
        print("viterbi already exists. rerun? y/n")
        while (x:=input()) not in ["y","n"]:
            pass
        if x == "y":
            run_cc_viterbi(config)
    else:
        run_cc_viterbi(config)

    viterbi_guess = load_viterbi_guess(config)

    true_state_seqs = get_true_state_seqs_from_true_MSA(config)

    compare_guess_to_true_state_seq(true_state_seqs, viterbi_guess)

    write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi_guess)

    eval_start_stop(config, viterbi_guess)
