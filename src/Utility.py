#!/usr/bin/env python3
import numpy as np
from Bio import SeqIO
import re
import time
from resource import getrusage
from resource import RUSAGE_SELF
import os
import json
import tensorflow as tf

##############################################################################
################################################################################
################################################################################
@tf.autograph.experimental.do_not_convert
def append_time_ram_stamp_to_file(description, bench_file_path, start = None, ):
    if not os.path.exists('/'.join(bench_file_path.split('/')[:-1])):
        os.system(f"mkdir -p {'/'.join(bench_file_path.split('/')[:-1])}")
    with open(bench_file_path, "a") as file:
        d = {}
        d["time"] = np.round(time.perf_counter(),3)
        d["time since passed start time"] = np.round(time.perf_counter() - start,3) if start != None else "no start passed"
        RAM = getrusage(RUSAGE_SELF).ru_maxrss
        d["RAM in kb"] = RAM
        d["RAM in GB"] = np.round(RAM/1024/1024,3)
        d["description"] = description

        json.dump(d, file)
        file.write("\n")
################################################################################
################################################################################
################################################################################

# ====> the code below may be outdated <=======================================
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
def find_indices_in_sparse_A_that_are_zero(config = None, \
                                           path_to_current_dense = None, \
                                           I_dense = None,
                                           A_dense = None,
                                           B_dense = None, \
                                           nCodons = None, \
                                           no_deletes = None, \
                                           no_inserts = None, \
                                           forced_gene_structure = None):
    if config == None:
        assert A_dense == None, "find_indices_in_sparse_A_that_are_zero: if config is None, so should A_dense"
    if A_dense == None:
        assert A_dense == None, "find_indices_in_sparse_A_that_are_zero: if A_dese is None, so should config"
    if config == None:
        assert path_to_current_dense != None, "if config and A_dense are None, you should specify a path_to_current_dense.json"

    if config != None:
        assert I_dense != None, "if config != None, then also pass I_dense"
        assert A_dense != None, "if config != None, then also pass A_dense"
        assert B_dense != None, "if config != None, then also pass B_dense"
    if A_dense != None or B_dense != None or I_dense != None:
        assert config != None, "if A_dense != None, then also pass config"

    if config == None: # if this method is manually called like: python3 -i Utility
        config = Config("main_programm")
        if nCodons != None:
            config.nCodons = nCodons
        if no_deletes != None:
            config.no_deletes = no_deletes
        if no_inserts != None:
            config.no_inserst = no_inserst
        if forced_gene_structure != None:
            config.forced_gene_structure = forced_gene_structure
        with open(path_to_current_dense, "r") as file:
            A_dense = np.array(json.load(file))
        print("finish code here kj345jh2kk23")
        exit()

    A_indices = get_indices_for_weights_for_A(config)
    A_indices += get_indices_for_constants_for_A(config)

    for index in A_indices:
        if A_dense[index] == 0:
            transition_from = state_id_to_description(index[0], config.nCodons)
            transition_to = state_id_to_description(index[1], config.nCodons)
            print(f"transition_from {transition_from} to {transition_to} is zero")

    B_indices = get_indices_for_weights_for_B(config)
    B_indices += get_indices_for_constants_for_B(config)

    for index in B_indices:
        # bc gets transposed
        if B_dense[index[1]][index[0]] == 0:
            state = state_id_to_description(index[0], config.nCodons)
            emission = id_to_higher_order_emission(index[1], config.alphabet_size, config.order)
            print(f"state {state} with emission {emission}")

    I_indices = get_indices_for_I(config)

    for index in I_indices:
        if I_dense[index[1], index[0]] == 0:
            state = state_id_to_description(index[0], config.nCodons)
            print(f"init state {state}")

################################################################################
################################################################################
################################################################################
def from_before_and_after_json_matrices_calc_diff_and_write_csv(config = None):
    import tensorflow as tf
    dir_before_fit = f"{config.current_run_dir}/before_fit_para"
    dir_after_fit  = f"{config.current_run_dir}/after_fit_para"
    print("started getting diff for a")
    out_dir  = f"{config.current_run_dir}/diff_after_and_before_matrices"

    with open(f"{dir_before_fit}/A.json", "r") as before_file:
        before = json.load(before_file)
    with open(f"{dir_after_fit}/A.json", "r") as after_file:
        after = json.load(after_file)
    diff = np.array(after) - np.array(before)
    diff = tf.constant(diff)
    # diff_2d_list = [[float(entry) for entry in col] for col in diff]
    config.model.A_as_dense_to_file(f"{out_dir}/A_diff.csv", "dummy_weights", with_description = True, A = diff)
    print("done with a")
    print("started getting diff for b")
    with open(f"{dir_before_fit}/B.json", "r") as before_file:
        before = json.load(before_file)
    with open(f"{dir_after_fit}/B.json", "r") as after_file:
        after = json.load(after_file)
    diff = np.array(after) - np.array(before)
    diff = tf.constant(diff)
    # diff_2d_list = [[float(entry) for entry in col] for col in diff]
    config.model.B_as_dense_to_file(f"{out_dir}/B_diff.csv", "dummy_weights", with_description = True, B = diff)
    print("done with b")

################################################################################
def get_A_for_viewing_parameters(before_fit = False, after_fit = False):
    assert not (before_fit and after_fit), "you cant take both"
    assert before_fit or after_fit, "you must choose one"
    config.init_weights_from_after_fit = after_fit
    config.init_weights_from_before_fit = before_fit
    assert config.AB == "dd", "pass -AB dd"
    from CgpHmmCell import CgpHmmCell
    cell = CgpHmmCell(config)
    not_used_parameter = 1
    cell.build(not_used_parameter)
    A = config.model.A(cell.A_kernel)
    return A
################################################################################
def view_parameters_in_A(index = None, description = None, before_fit = False):
    A = get_A_for_viewing_parameters(before_fit, not before_fit)
    if index != None:
        assert len(index) in [1,2], "check index"
        if len(index) == 2:
            print(f"A[{index}] = from {config.model.state_id_to_str(index[0])} to {config.model.state_id_to_str(index[1])} = {A[index]}")
            if A[index] == 0 and index in config.model.A_indices:
                print("the parameter was learned to be zero")
        if len(index) == 1:
            value_print_str = []
            for to_state, value in enumerate(A[index[0]]):
                value = A[[index[0], to_state]]
                if value != 0:
                    msg = f"A[{index[0],to_state}] = from {config.model.state_id_to_str(index[0])} to {config.model.state_id_to_str(to_state)} = {value}"
                    print(msg)
                    value_print_str.append((value, msg))

                if value == 0 and [index[0], to_state] in config.model.A_indices:
                    msg = f"A[{index[0],to_state}] = from {config.model.state_id_to_str(index[0])} to {config.model.state_id_to_str(to_state)} was parameter was learned to be zero"
                    print(msg)
                    value_print_str.append((value, msg))

            print("sorted:")
            for value, msg in sorted(value_print_str, reverse = True):
                print(msg)
################################################################################
def view_A_summary(how_many = 5, before_fit = False):
    A = get_A_for_viewing_parameters(before_fit, not before_fit)
    # deletes, inserts, codon continue, (insert continue doesnt seem to be a problem, yet)
    indices = {}
    indices["deletes"] = config.model.A_indices_deletes
    indices["enter_next_codon"] = config.model.A_indices_enter_next_codon
    indices["begin_inserts"] = config.model.A_indices_begin_inserts

    for key, value in indices.items():
        values = [(A[index], index, f"from {config.model.state_id_to_str(index[0])}, to {config.model.state_id_to_str(index[1])}") for index in value]
        print(key)
        for i, (value, index, index_str) in enumerate(sorted(values, reverse = True)):
            if i < how_many:
                print(f"{index}, {index_str} = {value}")
            if i == how_many:
                print("...")
            if i > len(values) - how_many:
                print(f"{index}, {index_str} = {value}")
################################################################################
def view_A_differences(how_many = 5):
    A_before = get_A_for_viewing_parameters(before_fit=True)
    A_after = get_A_for_viewing_parameters(after_fit=True)
    diff = []
    for index in config.model.A_indices:
        from_state = index[0]
        to_state = index[1]
        index_str = f"{config.model.state_id_to_str(from_state)}, {config.model.state_id_to_str(to_state)}"
        a_before = A_before[from_state, to_state]
        a_after = A_after[from_state, to_state]
        value = abs(a_before - a_after)
        diff.append((value, index, index_str, a_before, a_after))
    for i, (value, index, index_str, a_before, a_after) in enumerate(sorted(diff, reverse=True)):
        if i > how_many:
            break
        print(f"{index},\t{index_str},\tdiff = {a_after - a_before},\tbefore = {a_before},\tafter = {a_after}")
################################################################################
################################################################################
def get_B_for_viewing_parameters(before_fit = False, after_fit = False):
    assert not (before_fit and after_fit), "you cant take both"
    assert before_fit or after_fit, "you must choose one"
    config.init_weights_from_after_fit = after_fit
    config.init_weights_from_before_fit = before_fit
    assert config.AB == "dd", "pass -AB dd"
    from CgpHmmCell import CgpHmmCell
    cell = CgpHmmCell(config)
    not_used_parameter = 1
    cell.build(not_used_parameter)
    B = config.model.B(cell.B_kernel)
    return B
################################################################################
def view_parameters_in_B(state_id, how_many = 5, before_fit = False):
    B = get_B_for_viewing_parameters(before_fit, not before_fit)
    values = []
    for emission_id in range(len(B)):
        value = B[emission_id, state_id]
        values.append((value, (emission_id, state_id), f"{config.model.emission_id_to_str(emission_id)}, {config.model.state_id_to_str(state_id)}"))

    for i, (value, index, index_str) in enumerate(sorted(values, reverse=True)):
        if i > how_many:
            break
        print(f"{index}, {index_str} = {value}")

    if config.order > 0:
        given_prev_bases = {}
        for value, index, index_str in values:
            if index_str[:config.order] in given_prev_bases:
                given_prev_bases[index_str[:config.order]].append((value, index, index_str))
            else:
                given_prev_bases[index_str[:config.order]] = [(value, index, index_str)]

        given_prev_bases_with_std = []
        for last_bases, value_list in given_prev_bases.items():
            print(last_bases)
            data = np.array([entry[0] for entry in value_list])
            std = data.std()
            given_prev_bases_with_std.append((std, value_list))
            for value, index, index_str in value_list:
                print(f"{index}, {index_str} = {value}")
        print("the highest and lowest std")
        for i, (std, value_list) in enumerate(sorted(given_prev_bases_with_std, reverse=True)):
            if i < how_many:
                print("std =", std)
                for value, index, index_str in value_list:
                    print(f"{index}, {index_str} = {value}")
            if i == how_many:
                print("...")
            if i > len(given_prev_bases_with_std) - how_many - 3:# for X and "fill to multiple of 4" states
                print("std =", std)
                for value, index, index_str in value_list:
                    print(f"{index}, {index_str} = {value}")
################################################################################
def view_B_differences(how_many = 5):
    B_before = get_B_for_viewing_parameters(before_fit=True)
    B_after = get_B_for_viewing_parameters(after_fit=True)
    diff = []
    for index in config.model.B_indices:
        emission_id = index[0]
        state_id = index[1]
        index_str = f"{config.model.emission_id_to_str(emission_id)}, {config.model.state_id_to_str(state_id)}"
        b_before = B_before[emission_id, state_id]
        b_after = B_after[emission_id, state_id]
        value = abs(b_before - b_after)
        diff.append((value, index, index_str, b_before, b_after))
    for i, (value, index, index_str, b_before, b_after) in enumerate(sorted(diff, reverse=True)):
        if i > how_many:
            break
        print(f"{index},\t{index_str},\tdiff = {b_after - b_before},\tbefore = {b_before},\tafter = {b_after}")
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
def get_time_and_ram_from_bench_file(path):
    if not os.path.exists(path):
        print(f"file {path} does not exist")
        exit(1)

    with open(path, "r") as file:
        start_time = -1
        end_time = -1
        max_ram_in_gb = float("-inf")
        for line in file:
            # print("trying to laods =", line)
            data = json.loads(line.strip())
            if data["description"].startswith("Training.make_dataset() start"):
                start_time = data["time"]
            if data["description"].startswith("Training:model.fit() end"):
                end_time = data["time"]
            max_ram_in_gb = max(max_ram_in_gb, data["RAM in kb"]/1024/1024)
        assert start_time != -1, "start_time is not found"
        assert end_time != -1, "end_time is not found"
        y_time = end_time - start_time
    return y_time, max_ram_in_gb

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
    import tensorflow as tf
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
