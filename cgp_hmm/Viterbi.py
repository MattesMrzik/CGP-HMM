#!/usr/bin/env python3
import os
import pandas as pd
import json
import re
import time

def run_cc_viterbi(config):
    import multiprocessing
    start = time.perf_counter()
    print("starting viterbi")
    os.system(f"{config.src_path}/Viterbi " + str(config.nCodons) + " " + str(multiprocessing.cpu_count()-1))
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
    viterbi_file = open(f"{config.src_path}/output/{config.nCodons}codons/viterbi.json", "r")
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
            for i, line in enumerate(msa):
                seq_id = i - 1
                if seq_id < 0:
                    out.write(line)
                    continue
                else:
                    out.write(line) # ----TTATGTTCTAATCGGTT from useMSAgen trueMSA
                    out.write(expand_nice_states_str_to_fit_msa(state_seq_to_nice_str(true_state_seqs[seq_id]), line))
                    out.write(expand_nice_states_str_to_fit_msa(state_seq_to_nice_str(viterbi[seq_id]), line))
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

    start_stop = pd.read_csv(f"{config.src_path}/output/{config.nCodons}codons/out.start_stop_pos.{config.nCodons}codons.txt", sep=";", header=None)
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

if __name__ == "__main__":
    from Config import Config
    import numpy as np
    config = Config("main_programm_dont_interfere")

    I_out_path =f"{config.src_path}/output/{config.nCodons}codons/I.{config.nCodons}codons.csv"
    A_out_path =f"{config.src_path}/output/{config.nCodons}codons/A.{config.nCodons}codons.csv"
    B_out_path =f"{config.src_path}/output/{config.nCodons}codons/B.{config.nCodons}codons.csv"

    if not os.path.exists(A_out_path):
        # from cell.py
        path = f"{config.src_path}/output/{config.nCodons}codons/after_fit_kernels/"
        def read_weights_from_file(path):
            with open(f"{path}/I_kernel.json") as file:
                weights_I = np.array(json.load(file))
            with open(f"{path}/A_kernel.json") as file:
                weights_A = np.array(json.load(file))
            with open(f"{path}/B_kernel.json") as file:
                weights_B = np.array(json.load(file))
            return weights_I, weights_A, weights_B

        weights_I, weights_A, weights_B = read_weights_from_file(path)

        config.model.I_as_dense_to_json_file(I_out_path + ".json", weights_I)
        config.model.A_as_dense_to_json_file(A_out_path + ".json", weights_A)
        config.model.B_as_dense_to_json_file(B_out_path + ".json", weights_B)

    if os.path.exists(f"{config.src_path}/output/{config.nCodons}codons/viterbi.json"):
        print("viterbi already exists. If you want to rerun it, then delete viterbi.json")
    else:
        run_cc_viterbi(config)

    viterbi_guess = load_viterbi_guess(config)

    true_state_seqs = get_true_state_seqs_from_true_MSA(config)

    write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi_guess)

    eval_start_stop(config, viterbi_guess)
