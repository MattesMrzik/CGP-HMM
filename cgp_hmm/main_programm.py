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
        A_out_path =f"{config.src_path}/output/{config.nCodons}codons/A.{config.nCodons}codons.txt"
        # print(config.model.A_as_dense_to_str(cell.A_kernel, with_description = True))
        config.model.A_as_dense_to_file(A_out_path, A_kernel, with_description = False)
        config.model.A_as_dense_to_file(A_out_path + ".with_description", A_kernel, with_description = True)

        B_out_path =f"{config.src_path}/output/{config.nCodons}codons/B.{config.nCodons}codons.txt"
        # print(config.model.B_as_dense_to_str(cell.B_kernel, with_description = True))
        config.model.B_as_dense_to_file(B_out_path, B_kernel, with_description = False)
        config.model.B_as_dense_to_file(B_out_path + ".with_description", B_kernel, with_description = True)

    if config.nCodons < 10:
        config.model.export_to_dot_and_png(A_kernel, B_kernel)


    if config.run_viterbi:
        # running Viterbi
        run(f"{config.src_path}/Viterbi " + config.fasta_path + " " + str(config.nCodons))

        stats = {"start_not_found" : 0,\
                 "start_too_early" : 0,\
                 "start_correct" : 0,\
                 "start_too_late" : 0,\
                 "stop_not_found" : 0,\
                 "stop_too_early" : 0,\
                 "stop_correct" : 0,\
                 "stop_too_late" : 0}

        # comparing viterbi result with correct state seq
        with open(f"{config.src_path}/output/{config.nCodons}codons/viterbi.{config.nCodons}codons.csv", "r") as viterbi_file:
            with open(f"{config.src_path}/output/{config.nCodons}codons/out.start_stop_pos.{config.nCodons}codons.txt", "r") as start_stop_file:
                for v_line in viterbi_file:
                    try:
                        ss_line = start_stop_file.readline()
                    except:
                        print("ran out of line in :" + f"out.start_stop_pos.{config.nCodons}codons.txt")
                        quit(1)
                    if ss_line[:4] == ">seq" or len(ss_line) <= 1:
                        continue
                    true_start = int(ss_line.split(";")[0])
                    true_stop = int(ss_line.split(";")[1].strip())
                    try:
                        viterbi_start = v_line.split("\t").index("stA")
                    except:
                        viterbi_start = -1
                    try:
                        viterbi_stop = v_line.split("\t").index("st1")
                    except:
                        viterbi_stop = -1
                    # print(f"true_start = {true_start} vs viterbi_start = {viterbi_start}")
                    # print(f"true_stop = {true_stop} vs viterbi_stop = {viterbi_stop}")

                    if viterbi_start == -1:
                        stats["start_not_found"] += 1
                        if viterbi_stop != -1:
                            print("found stop but not start")
                            quit(1)
                    elif viterbi_start < true_start:
                        stats["start_too_early"] += 1
                    elif viterbi_start == true_start:
                        stats["start_correct"] += 1
                    else:
                        stats["start_too_late"] += 1

                    if viterbi_stop == -1:
                        stats["stop_not_found"] += 1
                    elif viterbi_stop < true_stop:
                        stats["stop_too_early"] += 1
                    elif viterbi_stop == true_stop:
                        stats["stop_correct"] += 1
                    else:
                        stats["stop_too_late"] += 1

        nSeqs = sum([v for v in stats.values()])/2 # div by 2 bc every seq appears twice in stats (in start and stop)

        with open(f"{config.src_path}/output/{cinfig.nCodons}codons/statistics.txt", "w") as file:
            for key, value in stats.items():
                file.write(key + "\t" + str(value/nSeqs) + "\n")



if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    main(config)
