#!/usr/bin/env python3
import argparse

def main(config):

    config.print()

    import tensorflow as tf
    import matplotlib.pyplot as plt
    from Training import fit_model
    import Utility
    import numpy as np
    from Bio import SeqIO
    import WriteData
    import re
    from Utility import run
    import json
    import pandas as pd

    model, history = fit_model(config)
    print("done fit_model()")
    # model.save("my_saved_model")

    # writng the loss history to file
    with open(f"{config.src_path}/output/{config.nCodons}codons/loss.log", "w") as file:
        for loss in history.history['loss']:
            file.write(str(loss))
            file.write("\n")

    plt.plot(history.history['loss'])
    plt.savefig(f"{config.src_path}/progress.png")

    I_kernel, A_kernel, B_kernel = model.get_weights()

    if config.write_parameters_after_fit:


        # print(config.model.A_as_dense_to_str(cell.A_kernel, with_description = True))
        A_out_path =f"{config.src_path}/output/{config.nCodons}codons/A.{config.nCodons}codons.csv"
        config.model.A_as_dense_to_file(A_out_path, A_kernel, with_description = False)
        config.model.A_as_dense_to_file(A_out_path + ".with_description.csv", A_kernel, with_description = True)

        B_out_path =f"{config.src_path}/output/{config.nCodons}codons/B.{config.nCodons}codons.csv"
        # print(config.model.B_as_dense_to_str(cell.B_kernel, with_description = True))
        config.model.B_as_dense_to_file(B_out_path, B_kernel, with_description = False)
        config.model.B_as_dense_to_file(B_out_path + ".with_description.csv", B_kernel, with_description = True)

        I_out_path =f"{config.src_path}/output/{config.nCodons}codons/I.{config.nCodons}codons.csv"
        config.model.I_as_dense_to_json_file(I_out_path + ".json", I_kernel)
        config.model.A_as_dense_to_json_file(A_out_path + ".json", A_kernel)
        config.model.B_as_dense_to_json_file(B_out_path + ".json", B_kernel)

    if config.nCodons < 10:
        config.model.export_to_dot_and_png(A_kernel, B_kernel)


    if config.run_viterbi:
        # write convert fasta file to json (not one hot)
        # see make_dataset in Training.py

        

        run(f"{config.src_path}/Viterbi " + str(config.nCodons))

        stats = {"start_not_found" : 0,\
                 "start_too_early" : 0,\
                 "start_correct" : 0,\
                 "start_too_late" : 0,\
                 "stop_not_found" : 0,\
                 "stop_too_early" : 0,\
                 "stop_correct" : 0,\
                 "stop_too_late" : 0}

        viterbi_file = open(f"{config.src_path}/output/{config.nCodons}codons/viterbi.json", "r")
        viterbi = json.load(viterbi_file)

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



if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    main(config)
