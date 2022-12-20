#!/usr/bin/env python3
import numpy as np
from Bio import SeqIO
import re
import time
from random import randint
import tensorflow as tf
from resource import getrusage
from resource import RUSAGE_SELF
from itertools import product
import os


np.set_printoptions(linewidth=200)
################################################################################
################################################################################
################################################################################
def get_indices_for_constants_from_transition_kernel_higher_order(config):
    nCodons = config.nCodons
    # from start a
    indices = [[1,2]]
    # from start t
    indices += [[2,3]]

    # first to second codon position
    indices += [[4 + i*3, 5 + i*3] for i in range(nCodons)]
    # second to third codon position
    indices += [[5 + i*3, 6 + i*3] for i in range(nCodons)]

    # inserts
    offset = 8 + 3*nCodons
    # begin inserts


    # first to second position in insert
    indices += [[offset + i*3, offset + 1 + i*3] for i in range(nCodons + 1)]
    # second to third position in insert
    indices += [[offset + 1 + i*3, offset + 2 + i*3] for i in range(nCodons + 1)]
    # ending an insert

    # stop T
    indices += [[4 + nCodons*3, 5 + nCodons*3]]

    # second to third position in stop
    indices += [[5 + nCodons*3, 6 + nCodons*3]]

    # stop -> ig 3'
    indices += [[6 + nCodons*3, 7 + nCodons*3]]


    index_of_terminal_1 = 8 + nCodons*3 + (nCodons + 1) *3
    indices += [[index_of_terminal_1, index_of_terminal_1]]

    return indices
################################################################################
def get_indices_for_weights_from_transition_kernel_higher_order(config): # no shared parameters
    nCodons = config.nCodons
    # from ig 5'
    indices = [[0,0], [0,1]]

    # enter codon
    indices += [[3 + i*3, 4 + i*3] for i in range(nCodons)]

    # begin inserts
    offset = 8 + 3*nCodons
    indices += [[3 + i*3, offset + i*3] for i in range(nCodons + 1)]
    # ending an insert
    indices += [[offset + 2 + i*3, 4 + i*3] for i in range(nCodons + 1)]
    # continuing an insert
    indices += [[offset + 2 + i*3, offset + i*3] for i in range(nCodons +1)]

    # exit last codon
    indices += [[3 + nCodons*3, 4 + nCodons*3]]

    # deletes
    i_delete = [3 + i*3 for i in range(nCodons) for j in range(nCodons-i)]
    j_delete = [4 + j*3 for i in range(1,nCodons+1) for j in range(i,nCodons+1)]
    indices += [[i,j] for i,j in zip(i_delete, j_delete)]

    # ig -> ig, terminal_1
    index_of_terminal_1 = 8 + nCodons*3 + (nCodons + 1) *3
    indices += [[7 + nCodons*3, 7 + nCodons*3], [7 + nCodons*3, index_of_terminal_1]]


    return indices
################################################################################
def get_indices_and_values_from_transition_kernel_higher_order(config, w):
    nCodons = config.nCodons
    k = 0
    # ig 5'
    indices = [[0,0], [0,1]]
    values = [1 - w[k], w[k]] # lieber sigmoid
    k += 1
    # start a
    indices += [[1,2]]
    values += [1]
    # start t
    indices += [[2,3]]
    values += [1]

    # enter codon
    indices += [[3 + i*3, 4 + i*3] for i in range(nCodons)]
    # print("values =", values)
    # print("w[k: k + nCodons] =", w[k: k + nCodons])
    values = tf.concat([values, w[k: k + nCodons]], axis = 0)
    k += nCodons
    # first to second codon position
    indices += [[4 + i*3, 5 + i*3] for i in range(nCodons)]
    values = tf.concat([values, [1] * nCodons], axis = 0)
    # second to third codon position
    indices += [[5 + i*3, 6 + i*3] for i in range(nCodons)]
    values = tf.concat([values, [1] * nCodons], axis = 0)

    # inserts
    offset = 8 + 3*nCodons
    # begin inserts
    use_inserts = True
    if use_inserts:
        indices += [[3 + i*3, offset + i*3] for i in range(nCodons + 1)]
        values = tf.concat([values, w[k: k + nCodons + 1]], axis = 0)
        k += nCodons + 1

    # exit last codon
    indices += [[3 + nCodons*3, 4 + nCodons*3]]
    values = tf.concat([values, [w[k]]], axis = 0)
    k += 1

    # first to second position in insert
    indices += [[offset + i*3, offset + 1 + i*3] for i in range(nCodons + 1)]
    values = tf.concat([values, [1] * (nCodons + 1)], axis = 0)
    # second to third position in insert
    indices += [[offset + 1 + i*3, offset + 2 + i*3] for i in range(nCodons + 1)]
    values = tf.concat([values, [1] * (nCodons + 1)], axis = 0)
    # ending an insert
    indices += [[offset + 2 + i*3, 4 + i*3] for i in range(nCodons + 1)]
    values = tf.concat([values, w[k: k + nCodons + 1]], axis = 0)

    # continuing an insert
    indices += [[offset + 2 + i*3, offset + i*3] for i in range(nCodons +1)]
    values = tf.concat([values, 1-w[k: k + nCodons +1]], axis = 0)
    k += nCodons + 1

    # deletes
    i_delete = [3 + i*3 for i in range(nCodons) for j in range(nCodons-i)]
    j_delete = [4 + j*3 for i in range(1,nCodons+1) for j in range(i,nCodons+1)]
    indices += [[i,j] for i,j in zip(i_delete, j_delete)]
    # print("deletes =", [1-w[k] * w[k]**((j-i)/3) for i,j in zip(i_delete, j_delete)])
    values = tf.concat([values, [1-w[k] * w[k]**int((j-i)/3) for i,j in zip(i_delete, j_delete)]], axis = 0)
    k += 1

    # stop T
    indices += [[4 + nCodons*3, 5 + nCodons*3]]
    values = tf.concat([values, [1]], axis = 0)

    # second to third position in stop
    indices += [[5 + nCodons*3, 6 + nCodons*3]]
    values = tf.concat([values, [1]], axis = 0)

    # stop -> ig 3'
    indices += [[6 + nCodons*3, 7 + nCodons*3]]
    values = tf.concat([values, [1]], axis = 0)

    # ig -> ig, terminal_1
    index_of_terminal_1 = 8 + nCodons*3 + (nCodons + 1) *3
    indices += [[7 + nCodons*3, 7 + nCodons*3], [7 + nCodons*3, index_of_terminal_1]]
    # values = tf.concat([values, [.5] * 2], axis = 0) # this parameter doesnt have to be learned (i think)
    # .5 can be any other number, since softmax(x,x) = [.5, .5]
    # but: TypeError: Cannot convert [0.5, 0.5] to EagerTensor of dtype int32   (todo)
    values = tf.concat([values, [1] * 2], axis = 0) # this parameter doesnt have to be learned (i think)


    # if self.order_transformed_input:
        # terminal -> terminal
    indices += [[index_of_terminal_1, index_of_terminal_1]]
    values = tf.concat([values, [1]], axis = 0)

    # not order transformed input
    # else:
    #     # terminal_1 -> terminal_1, a mix of true bases and X are emitted
    #     # terminal_1 -> terminal_2, only X are emitted
    #     indices += [[index_of_terminal_1, index_of_terminal_1], [index_of_terminal_1, index_of_terminal_1 +1]]
    #     values = tf.concat([values, [1] * 2], axis = 0)
    #
    #     # terminal_2 -> terminal_2
    #     indices += [[index_of_terminal_1 +1, index_of_terminal_1 +1]]
    #     values = tf.concat([values, [1]], axis = 0)



    return indices, values
################################################################################
def get_indices_for_config(config):
    config.state_id_description_list = get_state_id_description_list(config.nCodons)
    config.indices_for_weights_A = get_indices_for_weights_from_transition_kernel_higher_order(config)
    config.indices_for_constants_A = get_indices_for_constants_from_transition_kernel_higher_order(config)
    config.indices_for_A = config.indices_for_weights_A + config.indices_for_constants_A

    config.indices_for_weights_B = get_indices_for_weights_from_emission_kernel_higher_order(config)
    config.indices_for_constants_B = get_indices_for_constants_from_emission_kernel_higher_order(config)
    config.indices_for_B = config.indices_for_weights_B + config.indices_for_constants_B

    config.indices_for_I = get_indices_from_initial_kernel(config)
################################################################################
def print_config(config):
    # print("config =", config)
    s = "=====> config <====================================================\n"
    maxlen_key = max([len(key) for key in config.keys()])
    for key,value in config.items():
        s += (f"{' '*(maxlen_key-len(key))}{key}: {str(value)[:50]}{(' ..., shape: ' + str(tf.shape(value).numpy())) if len(str(value)) > 50 else ''}")
        s += "\n"
    s += "=====> config <===================================================="
    print(s)
################################################################################
def nucleotide_ambiguity_code_to_array(emission):
    # todo: somehow having this dict as self.code made it slower, why???
    code = {
        "A" : [0],
        "C" : [1],
        "G" : [2],
        "T" : [3],
        "Y" : [1,3],
        "R" : [0,2],
        "W" : [0,3],
        "S" : [1,2],
        "K" : [2,3],
        "M" : [0,1],
        "D" : [0,2,3],
        "V" : [0,1,2],
        "H" : [0,1,3],
        "B" : [1,2,3],
        "N" : [0,1,2,3],
        "X" : [5]
    }
    return code[emission]
################################################################################
def strip_or_pad_emission_with_n(config, ho_emission):
    return ["N"] * (config.order - len(ho_emission) + 1) + list(ho_emission)[- config.order - 1:]
################################################################################
def has_I_emission_after_base(config, emission):
    found_emission = False
    invalid_emission = False
    for i in range(config.order +1):
        if found_emission and emission[i] == config.alphabet_size:
            # print("not adding ", x)
            invalid_emission = True
            break
        if emission[i] != config.alphabet_size:
            found_emission = True
    return invalid_emission
################################################################################
def emission_is_stop_codon(ho_emission):
    stops = [[3,0,0],[3,0,2],[3,2,0]]
    if len(ho_emission) < 3:
        return False

    def same(a,b):
        for i in range(3):
            if a[i] != b[len(b) - 3 + i]:
                return False
        return True
    for stop in stops:
        if same(ho_emission, stop):
            return True
    return False
################################################################################
def state_is_third_pos_in_frame(config, state):
    des = state_id_to_description(state, config.nCodons, config.state_id_description_list)
    if des [-1] == "2" and des != "stop2" and des != "ter2":
        return True
    return False

def get_emissions_that_fit_ambiguity_mask(config, ho_mask, x_bases_must_preceed, state):

    # getting the allowd base emissions in each slot
    # ie "NNA" and x_bases_must_preceed = 2 -> [][0,1,2,3], [0,1,2,3], [0]]
    allowed_bases = [0] * (config.order + 1)
    for i, emission in enumerate(strip_or_pad_emission_with_n(config, ho_mask)):
        allowed_bases[i] = nucleotide_ambiguity_code_to_array(emission)
        if i < config.order - x_bases_must_preceed:
            allowed_bases[i] += [4] # initial emission symbol

    allowed_ho_emissions = []
    state_is_third_pos_in_frame_bool = state_is_third_pos_in_frame(config, state)
    for ho_emission in product(*allowed_bases):
        if not has_I_emission_after_base(config, ho_emission) \
        and not (state_is_third_pos_in_frame_bool and emission_is_stop_codon(ho_emission)):
            allowed_ho_emissions += [ho_emission]

    return allowed_ho_emissions


def get_indices_and_values_for_emission_higher_order_for_a_state(config, weights, \
                                                                 k, indices, \
                                                                 values, state, \
                                                                 mask, \
                                                                 x_bases_must_preceed, \
                                                                 trainable = True):
    # if self.order_transformed_input and emissions[-1] == "X":
    if mask[-1] == "X":
        indices += [[state, (config.alphabet_size + 1) ** (config.order +1)]]
        values[0] = tf.concat([values[0], [1]], axis = 0)
        return

    count_weights = 0
    for ho_emission in get_emissions_that_fit_ambiguity_mask(config, mask, x_bases_must_preceed, state):

        indices += [[state, higher_order_emission_to_id(ho_emission, config.alphabet_size, config.order)]]
        count_weights += 1

    if trainable:
        values[0] = tf.concat([values[0], weights[k[0]:k[0] + count_weights]], axis = 0)
        k[0] += count_weights
    else:
        values[0] = tf.concat([values[0], [1] * count_weights], axis = 0)

def get_indices_and_values_for_emission_higher_order_for_a_state_old_inputs(config, w, nCodons, alphabet_size):
    pass

def get_indices_and_values_from_emission_kernel_higher_order(config, w, nCodons, alphabet_size):
    indices = []
    values = [[]] # will contain one tensor at index 0, wrapped it in a list such that it can be passed by reference, ie such that it is mutable
    weights = w
    k = [0]

    # ig 5'
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,0,"N",0)
    # start a
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,1,"A",0)
    # start t
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,2,"AT",0)
    # start g
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,3,"ATG",2, trainable = False)
    # codon_11
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,4,"ATGN",2)
    # codon_12
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,5,"ATGNN",2)
    # all other codons
    for state in range(6, 6 + nCodons*3 -2):
        get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,state,"N",2)
    # stop
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,4 + nCodons*3,"T",config["order"])
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,5 + nCodons*3,"TA",config["order"])
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,5 + nCodons*3,"TG",config["order"])
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,6 + nCodons*3,"TAA",config["order"], trainable = False)
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,6 + nCodons*3,"TAG",config["order"], trainable = False)
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,6 + nCodons*3,"TGA",config["order"], trainable = False)
    # ig 3'
    get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,7 + nCodons*3,"N",config["order"])
    # inserts
    for state in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3):
        get_indices_and_values_for_emission_higher_order_for_a_state(config, weights,k,indices,values,state,"N",config["order"])

    get_indices_and_values_for_emission_higher_order_for_a_state(\
                 config, weights,k,indices,values,8 + nCodons*3 + (nCodons+1)*3,"X",config["order"])

    return indices, values[0]

def get_indices_for_emission_higher_order_for_a_state(config, \
                                                      indices, \
                                                      state, \
                                                      mask, \
                                                      x_bases_must_preceed):
    # if self.order_transformed_input and emissions[-1] == "X":
    if mask[-1] == "X":
        indices += [[state, (config.alphabet_size + 1) ** (config.order +1)]]
        return

    count_weights = 0
    for ho_emission in get_emissions_that_fit_ambiguity_mask(config, mask, x_bases_must_preceed, state):
        indices += [[state, higher_order_emission_to_id(ho_emission, config.alphabet_size, config.order)]]

def get_indices_for_weights_from_emission_kernel_higher_order(config):
    start = time.perf_counter()
    run_id = randint(0,100)
    append_time_ram_stamp_to_file(start, f"Cell.get_indices_for_weights_from_emission_kernel_higher_order() start   {run_id}", config.bench_path)
    nCodons = config.nCodons
    indices = []

    # ig 5'
    get_indices_for_emission_higher_order_for_a_state(config, indices,0,"N",0)
    # start a
    get_indices_for_emission_higher_order_for_a_state(config, indices,1,"A",0)
    # start t
    get_indices_for_emission_higher_order_for_a_state(config, indices,2,"AT",0)

    # codon_11
    get_indices_for_emission_higher_order_for_a_state(config, indices,4,"ATGN",2)
    # codon_12
    get_indices_for_emission_higher_order_for_a_state(config, indices,5,"ATGNN",2)
    # all other codons
    for state in range(6, 6 + nCodons*3 -2):
        get_indices_for_emission_higher_order_for_a_state(config, indices,state,"N",2)
    # stop
    get_indices_for_emission_higher_order_for_a_state(config, indices,4 + nCodons*3,"T", config.order)
    get_indices_for_emission_higher_order_for_a_state(config, indices,5 + nCodons*3,"TA", config.order)
    get_indices_for_emission_higher_order_for_a_state(config, indices,5 + nCodons*3,"TG", config.order)
    # ig 3'
    get_indices_for_emission_higher_order_for_a_state(config, indices,7 + nCodons*3,"N", config.order)
    # inserts
    for state in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3):
        get_indices_for_emission_higher_order_for_a_state(config, indices,state,"N", config.order)

    get_indices_for_emission_higher_order_for_a_state(\
                          config, indices,8 + nCodons*3 + (nCodons+1)*3,"X", config.order)

    append_time_ram_stamp_to_file(start, f"Cell.get_indices_for_weights_from_emission_kernel_higher_order() end   {run_id}", config.bench_path)

    return indices

def get_indices_for_constants_from_emission_kernel_higher_order(config):
    nCodons = config.nCodons
    indices = []

    get_indices_for_emission_higher_order_for_a_state(config, indices,3,"ATG",2)
    get_indices_for_emission_higher_order_for_a_state(config, indices,6 + nCodons*3,"TAA", config.order)
    get_indices_for_emission_higher_order_for_a_state(config, indices,6 + nCodons*3,"TAG", config.order)
    get_indices_for_emission_higher_order_for_a_state(config, indices,6 + nCodons*3,"TGA", config.order)

    return indices
################################################################################
def get_indices_from_initial_kernel(config):
    nCodons = config.nCodons
    # start and codons
    indices = [[i,0] for i in range(3 + nCodons*3)]
    # inserts
    indices += [[i,0] for i in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3)]

    return indices
################################################################################
################################################################################
################################################################################
def plot_time_against_ram(path):
    import matplotlib.pyplot as plt
    time = []
    ram = []
    description = []
    with open(path, "r") as file:
        for line in file:
            line = line.split("\t")
            start, duration, ram_peak, description_startend_id = line
            time.append(float(start))
            ram.append(float(ram_peak))
            description.append(description_startend_id.strip())

    start_time = min(time)
    time = [t - start_time for t in time]
    del start_time
    max_time = max(time)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #annotate
    max_ram_peak = max(ram)
    number_of_entries = len(time)
    plt.plot(time,ram, label = "time_and_ram")
    for i, xy in enumerate(zip(time, ram)):
        x = (max_time) * 9/10
        y = (max_ram_peak * .9) / number_of_entries * (i + 3)
        if i == 0 or time[i] > time[i-1] + (max_time)/100:
            ax.annotate(description[i], xy = xy, xytext = (x,y), textcoords='data', arrowprops = {"arrowstyle":"->"})

    plt.grid(alpha=0.4)
    plt.xlabel("time")
    plt.ylabel("ram")
    plt.legend();

    plt.show()

################################################################################
def plot_time_and_ram(path, bar = False, extrapolate = 1, degree = 3):
    import os
    import matplotlib.pyplot as plt

    def get_infos_from(type):
        files = os.listdir(path)
        files = sorted(files, key = lambda x: int(re.search("\d+", x).group(0)))
        max_n_codons = int(re.search("\d+", files[-1]).group(0))
        times = np.zeros(max_n_codons)
        ram_peaks = np.zeros(max_n_codons)
        for dir in files:
            if os.path.isdir(f"{path}/{dir}"):
                if os.path.exists(f"{path}/{dir}/{type}"):
                    with open(f"{path}/{dir}/{type}","r") as infile:
                        min_time = float("inf")
                        max_time = 0
                        ram_peak = 0
                        for line in infile:
                            description = line.split("\t")[3]

                            time = float(line.split("\t")[0])
                            min_time = min(min_time, time)
                            if re.search("Training.model.fit.. end", description):
                                max_time = max(max_time, time)

                            ram = float(line.split("\t")[2])
                            ram_peak = max(ram_peak, ram)
                        times[int(re.search("\d+", dir).group(0))-1] = max_time - min_time
                        ram_peaks[int(re.search("\d+", dir).group(0))-1] = ram_peak

        for i in range(max_n_codons-1,-1,-1):
            if times[i] == 0:
                max_n_codons -=1
        times = times[:max_n_codons]
        ram_peaks = ram_peaks[:max_n_codons]

        return {"max_n_codons":max_n_codons, "times":times, "max_ram_peaks":ram_peaks}



    fig = plt.figure(figsize=(12, 12))
    from itertools import product
    for id in range(6):
        # 3_TrueorderTransformedInput.log

        info = get_infos_from(f"{id}_TrueorderTransformedInput.log")
        print(f"{id}_TrueorderTransformedInput.log")
        max_n_codons = info["max_n_codons"]

        max_n_codons_extrapolate = max_n_codons * extrapolate

        times = info["times"]
        ram_peaks = info["max_ram_peaks"]

        ax1 = fig.add_subplot(320 + i +1)
        # plt.plot(, times, "bo-")
        # plt.plot(np.arange(max_n_codons), ram_peaks, "rx-")
        x = np.arange(1,max_n_codons +1)

        # should coefs all be positive? for a runtime s
        coef_times = np.polyfit(x,times, degree) # coef for x^degree is coef_times[0]
        coef_ram_peaks = np.polyfit(x,ram_peaks/1024, degree)
        print("coef_times:", coef_times)
        print("coef_ram_peaks:", coef_ram_peaks)

        color = 'tab:red'
        ax1.set_xlabel('ncodons')
        ax1.set_ylabel('time in sec', color=color)
        ax1.plot(x, times, "rx")

        x_extrapolate = np.arange(1,max_n_codons_extrapolate+1)
        y = [np.polyval(coef_times,x) for x in x_extrapolate]
        if extrapolate:
            ax1.plot(x_extrapolate, y, color = "tab:red")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('ram_peak in mb', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, ram_peaks/1024, "bx")
        y = [np.polyval(coef_ram_peaks ,x) for x in x_extrapolate]
        if extrapolate:
            ax2.plot(x_extrapolate, y, color = "tab:blue")
        ax2.tick_params(axis='y', labelcolor=color)

        title = ["AB sparse","A dense","B dense","AB dense","full matrices"][i]
        def x_to_power_of(exponent):
            if exponent == 0:
                return ""
            elif exponent == 1:
                return "x"
            else:
                return f"x^{exponent}"
        title = " + ".join([f"{round(cc,2)} {x_to_power_of(len(coef_times)-jj-1)}" for jj, cc in enumerate(coef_times)]) + title
        title += " " + " + ".join([f"{round(cc,2)} {x_to_power_of(len(coef_times)-jj-1)}"for jj, cc in enumerate(coef_ram_peaks)])
        ax1.title.set_text(title)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig("bench.png")

################################################################################
################################################################################
################################################################################
def get_state_id_description_list(nCodons):
    states = re.split(" ", "ig5' stA stT stG")
    states += ["c_" + str(i) + "," + str(j) for i in range(nCodons) for j in range(3)]
    states += re.split(" ", "stop1 stop2 stop3 ig3'")
    states += ["i_" + str(i) + "," + str(j) for i in range(nCodons+1) for j in range(3)]
    states += ["ter1", "ter2"]
    return states

def state_id_to_description(id, nCodons, state_id_description_list = None):
    if not state_id_description_list:
        state_id_description_list = get_state_id_description_list(nCodons)
    # print("nCodons =", nCodons)
    # print("id =", id)
    # print("state_id_to_descriptcation =", state_id_description_list)
    return state_id_description_list[id]

def description_to_state_id(des, nCodons, state_id_description_list = None):
    if not state_id_description_list:
        state_id_description_list = get_state_id_description_list(nCodons)
    try:
        return state_id_description_list.index(des)
    except:
        return -1

def higher_order_emission_to_id(emission, alphabet_size, order):
    # todo: emission 4,4,4 = I,I,I is not used, i might give this id to X
    # also 4,1,4 is not used
    if emission == "X" or emission ==  alphabet_size + 1 or emission == [alphabet_size+1]:
        return (alphabet_size + 1)**(order + 1)
    #                                 initial symbol
    return sum([base*(alphabet_size + 1)**(len(emission) - i -1) for i, base in enumerate(emission)])

def id_to_higher_order_emission(id, alphabet_size, order, as_string = False):
    emission = []
    if id == (alphabet_size + 1)**(order + 1):
        if as_string:
            return "X"
        else:
            return [alphabet_size +1]
    for i in range(order,0,-1):
        fits = int(id/((alphabet_size+1)**i))
        if fits < 1:
            emission += [0]
        else:
            id -= fits*((alphabet_size+1)**i)
            emission += [int(fits)]
    emission += [int(id)]
    if as_string:
        emission = "".join(["ACGTI"[base] for base in emission])
    return emission
################################################################################
def print_color_coded_fasta(fasta_path, start_stop_path):
    seq_dict = {}
    with open(fasta_path,"r") as file:
    #     for record in SeqIO.parse(file,"fasta"):
    #         id = int(re.search("(\d+)", record.id).group(1))
    #         print(id)
    #         seq_dict[id] = record.seq
        id = -1
        for line in file:
            if idm := re.search(">.*?(\d+)", line):
                id = int(idm.group(1))
                print("found id =", id)
            if seqm := re.search("([A|C|G|T]+)", line):
                seq_dict[id] = seqm.group(1)

    print(seq_dict)
    start_stop_dict = {}
    with open(start_stop_path, "r") as file:
        id = -1
        for line in file:
            if idm := re.search(">.*?(\d+)", line):
                id = int(idm.group(1))
            if xx := re.search("(\d+);(\d+);", line):
                if id == -1:

                    print("something went wrong in print_color_coded_fasta()")
                    exit(1)
                start_stop_dict[id] = (int(xx.group(1)), int(xx.group(2)))
    print(start_stop_dict)
    for id in sorted(list(seq_dict.keys())):
        print(id)
        print(seq_dict[id])
        z = np.zeros(len(seq_dict[id]))
        z[int(start_stop_dict[id][0])] = 1
        z[int(start_stop_dict[id][1])] = 2
        print("".join([" " if k == 0 else str(int(k)) for k in z]))
################################################################################
def view_current_inputs_txt(path):
    with open(path, "r") as file:
        for line in file:
            if len(line) > 1:
                data = list(map(int,re.sub("[\[\]]","", line).strip().split(" ")))
                data = np.argmax(data)
                data = id_to_higher_order_emission(data, 4 ,2)
                print("ACGTIX"[data[-1]], end = "")
            else:
                print()
        print()

################################################################################
################################################################################
################################################################################
# type is either I, A, B
def transform_json_to_csv(path, type, nCodons):
    import json
    with open(path, "r") as file:
        data = json.load(file)
    print("data =", data)
    if type == "I":
        data = data[0]
        with open(path + ".csv", "w") as file:
            for id, value in enumerate(data):
                file.write(state_id_to_description(id, nCodons))
                file.write(";")
                file.write(str(value))
                file.write("\n")
    elif type == "A":
        with open(path + ".csv", "w") as file:
            file.write(";")
            for state in range(len(data)):
                file.write(state_id_to_description(state, nCodons))
                file.write(";")
            file.write("\n")
            for id, row in enumerate(data):
                file.write(state_id_to_description(id, nCodons))
                file.write(";")
                for value in row:
                    file.write(str(value))
                    file.write(";")
                file.write("\n")
    elif type == "B":
        with open(path + ".csv", "w") as file:
            file.write(";")
            for state in range(len(data[0])):
                file.write(state_id_to_description(state, nCodons))
                file.write(";")
            file.write("\n")
            for id, row in enumerate(data):
                file.write(id_to_higher_order_emission(id, 4, 2, as_string = True))
                file.write(";")
                for value in row:
                    file.write(str(value))
                    file.write(";")
                file.write("\n")

################################################################################
def transform_verbose_txt_to_csv(path, nCodons):
    log = {}
    with open(path,"r") as file:
        for line in file:
            line = line.strip().split(";")
            # beginning of data entry
            if len(line) == 4:
                if line[2][0] != ">":
                    continue
                count = int(line[0])
                run_id = int(line[1])
                description = line[2][1:]
                data = [round(float(x),3) for x in re.sub("[\[\]]","", line[3]).split(" ")]
            else:
                data = [round(float(x),3) for x in re.sub("[\[\]]","", line[0]).split(" ")]
            if count not in log:
                e = {description : [data]}
                log[count] = {run_id : e}
            else:
                if run_id not in log[count]:
                    e = {description : [data]}
                    log[count][run_id] = e
                else:
                    if description not in log[count][run_id]:
                        log[count][run_id][description] = [data]
                    else:
                        log[count][run_id][description].append(data)
    # for count in log.keys():
    #     for id in log[count].keys():
    #         for description in log[count][id].keys():
    #             for data in log[count][id][description]:
    #                 print(count,id,description,data, sep = "\t")

    with open(path + ".csv","w") as file:
        import numpy as np
        sep = ";"
        decimal_seperator = ","
        file.write("A\n" + sep*2)
        for id in log[1]:
            for i in range(len(log[1][id]["A"])):
                file.write(state_id_to_description(i, nCodons))
                file.write(sep)
            file.write("\n")
            for row_id, data in enumerate(log[1][id]["A"]):
                file.write(sep)
                file.write(state_id_to_description(row_id, nCodons))
                file.write(sep)
                file.write(sep.join(list(map(str,data))).replace(".",decimal_seperator))
                file.write("\n")
            break
        file.write("B\n" + sep*2)
        for id in log[1]:
            for i in range(len(log[1][id]["A"])):
                file.write(state_id_to_description(i, nCodons))
                file.write(sep)
            file.write("\n")
            for row_id, data in enumerate(log[1][id]["B"]):
                alphabet_size = 4
                order = 2
                file.write("".join(["ACGTIT"[b] for b in id_to_higher_order_emission(row_id, alphabet_size, order)]))
                file.write(sep)
                file.write(str(row_id))
                file.write(sep)
                file.write(sep.join(list(map(str,data))).replace(".",decimal_seperator))
                file.write("\n")
            break

        for i in sorted(list(log)): # sort by count
            for id in log[i]:
                file.write(str(id)+"_")
                max_len = max([len(v) for k,v in log[i][id].items() if k not in ["A","B"]])
                inputs = sep.join("i") # since i will decode the one_hot encoding
                E = sep.join("E" * len(log[i][id]["E"][0]))
                R = sep.join("R" * len(log[i][id]["R"][0]))
                a = sep.join("a" * len(log[i][id]["forward"][0]))
                l = sep.join("l" * len(log[i][id]["loglik"][0]))
                file.write(f"{i}{sep}{inputs}{sep}{E}{sep}{R}{sep}{a}{sep}{l}\n")
                for row in range(max_len):
                    file.write(str(i))
                    file.write(sep)
                    try:
                        file.write(str(np.argmax(log[i][id]["inputs"][row])).replace(".",decimal_seperator))
                        file.write(sep)
                    except:
                        file.write(sep)
                        file.write(sep)
                    try:
                        file.write(sep.join(list(map(str, log[i][id]["E"][row]))).replace(".",decimal_seperator))
                        file.write(sep)
                    except:
                        file.write(sep * (len(log[i][id]["E"][0])-1))
                        file.write(sep)
                    try:
                        file.write(sep.join(list(map(str, log[i][id]["R"][row]))).replace(".",decimal_seperator))
                        file.write(sep)
                    except:
                        file.write(sep * (len(log[i][id]["R"][0])-1))
                        file.write(sep)
                    try:
                        file.write(sep.join(list(map(str, log[i][id]["forward"][row]))).replace(".",decimal_seperator))
                        file.write(sep)
                    except:
                        file.write(sep * (len(log[i][id]["forward"][0])-1))
                        file.write(sep)
                    try:
                        file.write(sep.join(list(map(str, log[i][id]["loglik"][row]))).replace(".",decimal_seperator))
                        file.write(sep)
                    except:
                        file.write(sep * (len(log[i][id]["loglik"][0])-1).replace(".",decimal_seperator))
                        file.write(sep)
                    file.write("\n")

################################################################################
################################################################################
################################################################################
def append_time_stamp_to_file(time, description, path):
    with open(path, "a") as file:
        file.write(f"{time}\t{description}\n")
def append_time_ram_stamp_to_file(start, description, path):
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        run(f"mkdir -p {'/'.join(path.split('/')[:-1])}")
    with open(path, "a") as file:
        s = [time.perf_counter(),
             time.perf_counter() - start,
             getrusage(RUSAGE_SELF).ru_maxrss]
        s = [str(round(x,5)) for x in s]
        s = "\t".join(s + [description + "\n"])
        file.write(s)

    # if re.search("init", description) or True:
    #     tf.print("in append time and ram stamp")
    #     with open(path,"r") as file:
    #         for line in file:
    #             tf.print(line.strip())
    #     tf.print("done with printing file")

def remove_old_bench_files(nCodons):

        output_path = f"bench/{nCodons}codons"

        run(f"rm {output_path}/callbackoutput_time_start.txt")
        run(f"rm {output_path}/callbackoutput_time_end.txt")
        run(f"rm {output_path}/callbackoutput_ram_start.txt")
        run(f"rm {output_path}/callbackoutput_ram_end.txt")
        run(f"rm {output_path}/stamps.log")
def remove_old_verbose_files(nCodons):

        output_path = f"verbose"

        run(f"rm {output_path}/{nCodons}codons.txt")

################################################################################
################################################################################
################################################################################
def tfprint(s):
    print("py: ", s)
    tf.print("tf: ", s)
################################################################################
################################################################################
################################################################################
def run(command):
    # os.system(f"echo '\033[96mrunning -> {command} \033[00m'")
    os.system(f"echo running: {command}")
    os.system(command)
    # import subprocess
    # import random
    # import time
    # random_id = random.randint(0,1000000)
    # with open(f"temporary_script_file.{random_id}.sh","w") as file:
    #     file.write("#!/bin/bash\n")
    #     file.write(f"echo \033[91m running: \"{command}\" \033[00m\n")
    #     file.write(command)
    #
    # subprocess.Popen(f"chmod +x temporary_script_file.{random_id}.sh".split(" ")).wait()
    # subprocess.Popen(f"./temporary_script_file.{random_id}.sh").wait()
    # subprocess.Popen(f"rm temporary_script_file.{random_id}.sh".split(" ")).wait()
################################################################################
################################################################################
################################################################################
def generate_state_emission_seqs(a,b,n,l, a0 = [], one_hot = False):

    state_space_size = len(a)
    emission_space_size = len(b[0])

    states = 0
    emissions = 0

    def loaded_dice(faces, p):
        return np.argmax(np.random.multinomial(1,p))

    # todo just use else case, this can be converted by tf.one_hot
    if one_hot:
        states = np.zeros((n,l,state_space_size), dtype = np.int64)
        emissions = np.zeros((n,l,emission_space_size), dtype = np.int64)
        for i in range(n):
            states[i,0,0 if len(a0) == 0 else loaded_dice(state_space_size, a0)] = 1
            emissions[i,0, loaded_dice(emission_space_size, b[np.argmax(states[i,0,:])])] = 1
            for j in range(1,l):
                states[i,j, loaded_dice(state_space_size, a[np.argmax(states[i,j-1])])] = 1
                emissions[i,j, loaded_dice(emission_space_size, b[np.argmax(states[i,j-1])])] = 1
            # emssions.write(seq + "\n")
            # states.write(">id" + str(i) + "\n")
            # states.write(state_seq + "\n")
    else:
        states = np.zeros((n,l), dtype = np.int64)
        emissions = np.zeros((n,l), dtype = np.int64)
        for i in range(n):
            states[i,0] = 0 if len(a0) == 0 else loaded_dice(state_space_size, a0)
            emissions[i,0] = loaded_dice(emission_space_size, b[states[i,0]])
            for j in range(1,l):
                states[i,j] = loaded_dice(state_space_size, a[states[i,j-1]])
                emissions[i,j] = loaded_dice(emission_space_size, b[states[i,j-1]])

    return states, emissions
################################################################################
################################################################################
################################################################################
def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

################################################################################
################################################################################
################################################################################

# forward = P_theta(Y)
def forward(a,b,y, a0 = []):
    num_states = len(a)
    alpha = np.zeros((num_states,len(y)))
    if len(a0) == 0:
        alpha[0,0] = b[0,y[0]] # one must start in the first state
    else:
        alpha[:,0] = a0 * b[:,y[0]]

    for i in range(1,len(y)):
        for q in range(num_states):
            alpha[q,i]=b[q,y[i]]*sum([a[q_,q]*alpha[q_,i-1] for q_ in range(num_states)])
    #P(Y=y)
    p = sum([alpha[q,len(y)-1] for q in range(num_states)])
    return alpha, p

def forward_log_version(a,b,y, a0 = []):
    num_states = len(a)
    alpha = np.zeros((num_states,len(y)))
    if len(a0) == 0:
        alpha[:,0] = float("-inf")
        alpha[0,0] = np.log(b[0,y[0]]) # one must start in the first state
    else:
        alpha[:,0] = np.log(a0 * b[:,y[0]])

    for i in range(1,len(y)):
        for q in range(num_states):
            alpha[q,i]=np.log(b[q,y[i]]) + np.log(sum([a[q_,q] * np.exp(alpha[q_,i-1]) for q_ in range(num_states)]))
    #P(Y=y)
    p = np.log(sum([np.exp(alpha[q,len(y)-1]) for q in range(num_states)]))
    return alpha, p

def forward_felix_version(a,b,y, a0 = []):
    b = tf.transpose(b)
    num_states = len(a)
    alpha = np.zeros((num_states,len(y)))
    if len(a0) == 0:
        alpha[0,0] = b[0,y[0]] # one must start in the first state
    else:
        alpha[:,0] = a0 * b[:,y[0]]
    z = np.zeros(len(y))
    z[0] = sum([alpha[q_,0] for q_ in range(num_states)])
    for i in range(1,len(y)):
        for q in range(num_states):

            try:
                asdf = b[q,y[i]] * sum([a[q_,q] * alpha[q_,i-1]/z[i-1] for q_ in range(num_states)])
            except:
                print("y[i] =", y[i])
                print("b[q,y[i]] =", b[q,y[i]])
                for q_ in range(num_states):
                    print("q_ =", q_)
                    # print("a[q_,q] =", a[q_,q])
                    print("alpha[q_,i-1] =", alpha[q_,i-1])
                    print("z[i-1] =",z[i-1])

            alpha[q,i] = b[q,y[i]] * sum([a[q_,q] * alpha[q_,i-1]/z[i-1] for q_ in range(num_states)])
        z[i] = sum([alpha[q_,i] for q_ in range(num_states)])
    #P(Y=y)
    return alpha, z

# def forward_felix_version_ordertransformedinput(a,b,y, a0 = []):
#     # y is asumed to be one hot
#     num_states = len(a)
#     alpha = np.zeros((num_states,len(y)))
#     if len(a0) == 0:
#         alpha[0,0] = b[0,y[0]] # one must start in the first state
#     else:
#         alpha[:,0] = a0 * b[:,y[0]]
#     z = np.zeros(len(y))
#     z[0] = sum([alpha[q_,0] for q_ in range(num_states)])
#     for i in range(1,len(y)):
#         for q in range(num_states):
#             alpha[q,i] = b[q,y[i]] * sum([a[q_,q] * alpha[q_,i-1]/z[i-1] for q_ in range(num_states)])
#         z[i] = sum([alpha[q_,i] for q_ in range(num_states)])
#     #P(Y=y)
#     return alpha, z

def brute_force_P_of_Y(a,b,y, a0 = []):
    from itertools import product

    P_of_Y = 0

    n = len(y) - 1 if len(a0) == 0 else len(y)
    for x in product(list(range(len(a))), repeat = n):

        if len(a0) == 0:
            P_of_X_Y = 1 * b[0,y[0]]
        else:
            P_of_X_Y = a0[x[0]] * b[x[0],y[0]]
        for i in range(n - len(y) + 1, n):
            P_of_X_Y *= a[x[i-1],x[i]] * b[x[i],y[i]]
        P_of_Y += P_of_X_Y
    return P_of_Y #  dont need log version since underflow only happends with long sequnces, these here are short anyways since length is capped by runtime

################################################################################
################################################################################
################################################################################

def P_of_Y_given_X(a,b,x):
    P = 1
    for q in x:
        P *= b[q]
    return P

def P_of_X_i_is_q_given_Y(a,b,y,q,cca0 = []):
    pass

def P_of_X_Y(a,b,x,y, a0 = []):
    if len(a0) == 0 and x[0] != 0:
        return 0
    p = b[0, y[0]] if len(a0) == 0 else a0[x[0]] * b[x[0], y[0]]
    for i in range(1,len(y)):
        p *= a[x[i-1], x[i]]
        p *= b[x[i], y[i]]
    return p

def P_of_X_Y_log_version(a,b,x,y, a0 =[]):
    if len(a0) == 0 and x[0] != 0:
        return float("-inf")
    p = np.log(b[0, y[0]]) if len(a0) == 0 else np.log(a0[x[0]] * b[x[0], y[0]])
    for i in range(1,len(y)):
        p += np.log(a[x[i-1], x[i]])
        p += np.log(b[x[i], y[i]])
    return p

################################################################################
################################################################################
################################################################################
# argmax_x: P_theta(x|y)

# todo: implement this in c++
# maybe write an api for it
def viterbi_log_version_higher_order(a,b,i,y):
    import tensorflow as tf
    nStates = len(a)
    n = len(y)
    order = len(tf.shape(b)) -1 -1 # one for state, the other for current emission
    y_old = [4] * order # oldest to newest

    g = np.log(np.zeros((nStates, n))) # todo: i think it dont need log, since ive got I

    # for every state, at seq pos 0
    for state in range(nStates):
        index = [state] + y_old + [y[0]]
        g[state, 0] = np.log(i[state,0] * b[index])

    # todo only save last col of gamma, for backtracking recompute
    for i in range(1, n):
        print(str(i) + "/" + str(n), end = "\r")
        y_old = y_old[1:] + [y[i-1]]
        for q in range(nStates):
            # todo: change this to a for loop, and save current max, may impove runtime a bit
            # todo: can compute in parallel for different states
            m = max([np.log(a[state, q]) + g[state, i-1] for state in range(nStates)])
            index = [q] + y_old + [y[i]]
            g[q,i] = np.log(b[index]) + m
    # backtracking
    x = np.zeros(n, dtype = np.int32)
    x[n-1] = np.argmax(g[:,n-1])
    for i in range(n-2, -1, -1):
        x[i] = np.argmax(np.log(a[:,x[i+1]]) + g[:,i])
    return(x)


def viterbi_log_version(a, b, y, a0 = []):
    n = len(y)
    g = np.log(np.zeros((len(a), n)))
    if len(a0) == 0:
        g[0,0] = np.log(b[0, y[0]])
    else:
        g[:,0] = np.log(a0 * b[:, y[0]])
        # print("a0 =", a0)
        # print("b", b[:, y[0]])
        # print("g", g[:,0])
    # todo only save last col of gamma, for backtracking recompute
    for i in range(1, n):
        for q in range(len(a)):
            m = max(np.log(a[:,q]) + g[:,i-1])
            g[q,i] = np.log(b[q, y[i]]) + m
    # back tracking
    x = np.zeros(n, dtype = np.int32)
    x[n-1] = np.argmax(g[:,n-1])
    for i in range(n-2, -1, -1):
        x[i] = np.argmax(np.log(a[:,x[i+1]]) + g[:,i])
    return(x)

def brute_force_viterbi_log_version(a,b,y,a0 = []):
    from itertools import product
    max = float("-inf")
    arg_max = 0
    n = len(y) - 1 if len(a0) == 0 else len(y)
    for guess in product(list(range(len(a))), repeat = n):
        guess = [0] + list(guess) if len(a0) == 0 else guess
        p = P_of_X_Y_log_version(a,b,guess,y, a0)
        if p > max:
            max = p
            arg_max = guess
    return np.array(arg_max)


################################################################################
################################################################################
################################################################################
def create_layer_without_recursive_call():
    with open("CgpHmmLayer.py", "r") as layer:
        with open("CgpHmmLayer_non_recursive.py", "w") as copy:
            in_recursive_call = False
            for line in layer:
                if re.search("# do not change this line", line):
                    in_recursive_call = not in_recursive_call
                elif not in_recursive_call and not re.search("CgpHmmLayer_non_recursive", line):
                    line = re.sub("CgpHmmLayer","CgpHmmLayer_non_recursive",line)
                    copy.write(line)
################################################################################
################################################################################
################################################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='pass "-f filename" to transfrom verbose output of E,R,alpha to csv\nand "-c [int]" for nCodons\nmust have -b option when running main_programm')
    parser.add_argument('-f', '--filename',help='pass "-f filename" to transfrom verbose output of E,R,alpha to csv')
    parser.add_argument('-c', '--nCodons',help ='nCodons')
    parser.add_argument('-p', action='store_true', help ="plot bench folder")
    parser.add_argument('--create_layer_without_recursive_call', action='store_true', help = 'create_layer_without_recursive_call, create a copy of CgpHmmLayer but without the recursive call which is used for calculating the gradient')

    args = parser.parse_args()
    if args.p:
        plot_time_and_ram("bench", extrapolate = 1, degree = 2)
    elif args.nCodons and args.filename:
        if args.filename and args.nCodons:
            transform_verbose_txt_to_csv(args.filename, int(args.nCodons))
    elif args.create_layer_without_recursive_call:
        create_layer_without_recursive_call()
