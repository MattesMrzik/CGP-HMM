#!/usr/bin/env python3

import os

nCodons = 50
epochs = 1
steps = 1
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

#next dont train. or also use init_weights_from_before_fit on the last 2 runs
