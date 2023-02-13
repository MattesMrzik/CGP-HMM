#!/usr/bin/env python3

import os
import json

nCodons = 1
epochs = 4
steps = 4
epsilon_l = 1e-10
epsilon_E = 1e-10
epsilon_R = 1e-10

loglikes = []

def add_to_loglikes(s):
    loglikes_path = f"output/{nCodons}codons/loss.log"
    loglikes.append(s)
    loglikes.append(open(loglikes_path).readlines())
    loglikes.append(" ")

def call(s):
    os.system(s)
    add_to_loglikes(s)

only_train_with_epsilon = False

if only_train_with_epsilon:
    call(f"./main_programm.py --epoch {1} --step {1} -c {nCodons} --batch_be")
    # only to test whether writing and readning weights worked
    # if loglikes are the same as in the first run
    call(f"./main_programm.py --epoch {1} --step {1} -c {nCodons} --init_weights_from_before_fit --dont_generate")
    call(f"./main_programm.py --epoch {1} --step {1} -c {nCodons} --init_weights_from_before_fit --dont_generate --log --epsilon_l {epsilon_l} --epsilon_E {epsilon_E} --epsilon_R {epsilon_R}")
    call(f"./main_programm.py --epoch {epochs} --step {steps} -c {nCodons} --init_weights_from_before_fit --dont_generate --log --epsilon_l {epsilon_l} --epsilon_E {epsilon_E} --epsilon_R {epsilon_R} --write_parameters_after_fit")
    call(f"./main_programm.py --epoch {1} --step {1} -c {nCodons} --init_weights_from_after_fit --dont_generate")
    call(f"./main_programm.py --epoch {1} --step {1} -c {nCodons} --init_weights_from_after_fit --dont_generate --log --epsilon_l {epsilon_l} --epsilon_E {epsilon_E} --epsilon_R {epsilon_R}")

    for l in loglikes:
        print(l)

if not only_train_with_epsilon:
    call(f"./main_programm.py --epoch {1} --step {1} -c {nCodons} --batch_be")
    call(f"./main_programm.py --epoch {epochs} --step {steps} -c {nCodons} --init_weights_from_before_fit --dont_generate --write_parameters_after_fit --write_matrices_after_fit --log --epsilon_l {epsilon_l} --epsilon_E {epsilon_E} --epsilon_R {epsilon_R}")
    I_kernel_log = json.load(open(f"output/{nCodons}codons/after_fit_kernels/I_kernel.json","r"))
    A_kernel_log = json.load(open(f"output/{nCodons}codons/after_fit_kernels/A_kernel.json","r"))
    B_kernel_log = json.load(open(f"output/{nCodons}codons/after_fit_kernels/B_kernel.json","r"))

    I_log = json.load(open(f"output/{nCodons}codons/I.{nCodons}codons.csv.json","r"))
    A_log = json.load(open(f"output/{nCodons}codons/A.{nCodons}codons.csv.json","r"))
    B_log = json.load(open(f"output/{nCodons}codons/B.{nCodons}codons.csv.json","r"))

    call(f"./main_programm.py --epoch {epochs} --step {steps} -c {nCodons} --init_weights_from_before_fit --dont_generate --write_parameters_after_fit --write_matrices_after_fit ")
    I_kernel = json.load(open(f"output/{nCodons}codons/after_fit_kernels/I_kernel.json","r"))
    A_kernel = json.load(open(f"output/{nCodons}codons/after_fit_kernels/A_kernel.json","r"))
    B_kernel = json.load(open(f"output/{nCodons}codons/after_fit_kernels/B_kernel.json","r"))

    I = json.load(open(f"output/{nCodons}codons/I.{nCodons}codons.csv.json","r"))
    A = json.load(open(f"output/{nCodons}codons/A.{nCodons}codons.csv.json","r"))
    B = json.load(open(f"output/{nCodons}codons/B.{nCodons}codons.csv.json","r"))

    ############################################################################
    max_i_kernel = 0
    for i in range(len(I_kernel)):
        max_i_kernel = max_i_kernel if abs(I_kernel[i] - I_kernel_log[i]) < max_i_kernel else abs(I_kernel[i] - I_kernel_log[i])
    print("max_i_kernel =", max_i_kernel)

    max_i = 0
    for i in range(len(I[0])):
        max_i = max_i if abs(I[0][i] - I_log[0][i]) < max_i else abs(I[0][i] - I_log[0][i])
    print("max_i =", max_i)

    ############################################################################
    max_a_kernel = 0
    for i in range(len(A_kernel)):
        max_a_kernel = max_a_kernel if abs(A_kernel[i] - A_kernel_log[i]) < max_a_kernel else abs(A_kernel[i] - A_kernel_log[i])
    print("max_a_kernel =", max_a_kernel)

    max_a = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            max_a = max_a if abs(A[i][j] - A_log[i][j]) < max_a else abs(A[i][j] - A_log[i][j])
    print("max_a =", max_a)

    ############################################################################
    max_b_kernel = 0
    for i in range(len(B_kernel)):
        max_b_kernel = max_b_kernel if abs(B_kernel[i] - B_kernel_log[i]) < max_b_kernel else abs(B_kernel[i] - B_kernel_log[i])
    print("max_b_kernel =", max_b_kernel)

    max_b = 0
    for i in range(len(B)):
        for j in range(len(B[i])):
            max_b = max_b if abs(B[i][j] - B_log[i][j]) < max_b else abs(B[i][j] - B_log[i][j])
    print("max_b =", max_b)
    ############################################################################
