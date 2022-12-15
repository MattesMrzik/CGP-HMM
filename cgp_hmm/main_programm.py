#!/usr/bin/env python3
import argparse

def main(config):

    config.print()

    import tensorflow as tf
    import matplotlib.pyplot as plt
    from Training import fit_model
    from Training import make_dataset
    import Utility
    import numpy as np
    from Bio import SeqIO
    import WriteData
    from tensorflow.python.client import device_lib

    from Utility import remove_old_bench_files
    from Utility import remove_old_verbose_files

    from CgpHmmCell import CgpHmmCell

    from itertools import product
    codons = []
    for codon in product("ACGT", repeat = 3):
        codon = "".join(codon)
        if codon not in ["TAA", "TGA", "TAG"]:
            codons += [codon]

    if not config.dont_generate_new_seqs:
        if config.use_simple_seq_gen:
            num_seqs = 100
            seqs = {}
            with open(config.fasta_path, "w") as file:
                genlen = 3 * nCodons # ATG and STOP are not on gene
                seqlen = genlen * args.l
                seqlen += 6 # start and stop codon
                seqlen += 2 # ig states
                max_flanklen = (seqlen - genlen )//2
                low = max_flanklen -1 if config.dont_strip_flanks else 1

                for seq_id in range(num_seqs):

                    ig5 = "".join(np.random.choice(["A","C","G","T"], np.random.randint(low, max_flanklen))) # TODO: also check if low = 2
                    atg = "ATG"
                    # coding = "".join(np.random.choice(["A","C","G","T"], config["nCodons"] * 3))
                    coding = "".join(np.random.choice(codons, config["nCodons"]))
                    stop = np.random.choice(["TAA","TGA","TAG"])
                    ig3 = "".join(np.random.choice(["A","C","G","T"], np.random.randint(low, max_flanklen)))

                    seqs[f">my_generated_seq{seq_id}"] = ig5 + atg + coding + stop + ig3
                for key, value in seqs.items():
                    file.write(key + "\n")
                    file.write(value + "\n")
        else:
            from Utility import run
            run(f"python3 {config.src_path}/useMSAgen.py -c {config.nCodons} \
                          {'-n 4'} \
                          {'-l' + str(config.l)} \
                          {'-cd ' + str(config.coding_dist) if config.coding_dist else ''} \
                          {'-ncd ' + str(config.noncoding_dist) if config.noncoding_dist else ''}\
                          {'--dont_strip_flanks' if config.dont_strip_flanks else ''} \
                          {'-p ' + config.src_path if config.src_path else ''}" )


    model, history = fit_model(config)
    print("done fit_model()")
    # model.save("my_saved_model")

    with open(f"{config.src_path}/output/{config.nCodons}codons/loss.log", "w") as file:
        for loss in history.history['loss']:
            file.write(str(loss))
            file.write("\n")

    plt.plot(history.history['loss'])
    plt.savefig(f"{config.src_path}/progress.png")

    cell = CgpHmmCell(config)
    cell.init_kernel = model.get_weights()[0]
    cell.transition_kernel = model.get_weights()[1]
    cell.emission_kernel = model.get_weights()[2]

    def printA():
        global cell, model
        print(".\t", end ="")
        for state in range(cell.state_size[0]):
            print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
        print()
        for state in range(cell.state_size[0]):
            print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
            for goal_state in cell.A[state]:
                print((tf.math.round(goal_state*100)/100).numpy(), end = "\t")
            print()
    # printA()

    def printB():
        global cell, model
        for state in range(len(cell.B)):
            tf.print(Utility.state_id_to_description(state, cell.nCodons))
            tf.print(tf.math.round(cell.B[state]*100).numpy()/100, summarize = -1)
            tf.print("---------------------------------------------")
    # printB()

    def printI():
        global cell, model
        for state in range(len(cell.I)):
            print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
            print(tf.math.round(cell.I[state,0]*100).numpy()/100)
    # printI()

    #  bc with call type 4 A_dense fails
    if not config.call_type == 4 and config.run_viterbi:
        WriteData.write_to_file(cell.A_dense, f"{config.src_path}/output/{config.nCodons}codons/A.{config.nCodons}codons.txt")
        WriteData.write_to_file(tf.transpose(cell.B_dense), f"{config.src_path}/output/{config.nCodons}codons/B.{config.nCodons}codons.txt")
        WriteData.write_order_transformed_B_to_csv(cell.B_dense, f"{config.src_path}/output/{config.nCodons}codons/B.{config.nCodons}codons.csv", config.order, config.nCodons)

        WriteData.write_to_file(cell.I_dense, f"{config.src_path}/output/{config.nCodons}codons/I.{config.nCodons}codons.txt")

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

    if config.nCodons < 10:
        run(f"python3 {config.src_path}/Visualize.py -c {config.nCodons} -o {config.order} {'-t' if config.order_transformed_input else ''}")

if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    main(config)
