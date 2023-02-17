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
    import json
    import pandas as pd
    import time
    import os
    import datetime

    model, history = fit_model(config)
    print("done fit_model()")
    # model.save("my_saved_model")

    # writng the loss history to file
    with open(f"{config.src_path}/output/{config.nCodons}codons/loss.log", "w") as file:
        for loss in history.history['loss']:
            file.write(str(loss) + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            file.write("\n")

    plt.plot(history.history['loss'])
    plt.savefig(f"{config.src_path}/progress.png")

    I_kernel, A_kernel, B_kernel = model.get_weights()

    if config.write_matrices_after_fit:

        start = time.perf_counter()
        print("starting to write model")

        I_out_path =f"{config.src_path}/output/{config.nCodons}codons/I.{config.nCodons}codons.csv"
        A_out_path =f"{config.src_path}/output/{config.nCodons}codons/A.{config.nCodons}codons.csv"
        B_out_path =f"{config.src_path}/output/{config.nCodons}codons/B.{config.nCodons}codons.csv"

        # print(config.model.A_as_dense_to_str(cell.A_kernel, with_description = True))
        if config.nCodons < 20:
            config.model.A_as_dense_to_file(A_out_path, A_kernel, with_description = False)
            config.model.A_as_dense_to_file(A_out_path + ".with_description.csv", A_kernel, with_description = True)

        # print(config.model.B_as_dense_to_str(cell.B_kernel, with_description = True))
            config.model.B_as_dense_to_file(B_out_path, B_kernel, with_description = False)
            config.model.B_as_dense_to_file(B_out_path + ".with_description.csv", B_kernel, with_description = True)

        config.model.I_as_dense_to_json_file(I_out_path + ".json", I_kernel)
        config.model.A_as_dense_to_json_file(A_out_path + ".json", A_kernel)
        config.model.B_as_dense_to_json_file(B_out_path + ".json", B_kernel)

        print("done write model. it took ", time.perf_counter() - start)

    if config.write_parameters_after_fit:
        path = f"{config.src_path}/output/{config.nCodons}codons/after_fit_kernels"
        model.get_layer("cgp_hmm_layer").C.write_weights_to_file(path)

    if config.nCodons < 10:
        config.model.export_to_dot_and_png(A_kernel, B_kernel)


    if config.viterbi:
        # write convert fasta file to json (not one hot)
        # see make_dataset in Training.py

        import Viterbi

        Viterbi.run_cc_viterbi(config)

        viterbi_guess = Viterbi.load_viterbi_guess(config)

        true_state_seqs = Viterbi.get_true_state_seqs_from_true_MSA(config)

        Viterbi.write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi_guess)

        Viterbi.eval_start_stop(config, viterbi_guess)



if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    main(config)
