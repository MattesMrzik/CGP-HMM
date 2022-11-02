#!/usr/bin/env python3
import numpy as np
from Bio import SeqIO
import re
import time
import tensorflow as tf
from resource import getrusage
from resource import RUSAGE_SELF

np.set_printoptions(linewidth=200)
########################################################################
########################################################################
########################################################################
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
def plot_time_and_ram(path, bar = False):
    import os
    import matplotlib.pyplot as plt
    files = os.listdir(path)
    files = sorted(files, key = lambda x: int(re.search("\d+", x).group(0)))
    max_n_codons = int(re.search("\d+", files[-1]).group(0))
    times = np.zeros(max_n_codons)
    ram_peaks = np.zeros(max_n_codons)
    for file in files:
        if os.path.isdir(f"{path}/{file}"):
            with open(f"{path}/{file}/stamps.log","r") as infile:
                min_time = float("inf")
                max_time = 0
                ram_peak = 0
                for line in infile:
                    time = float(line.split("\t")[0])
                    min_time = min(min_time, time)
                    max_time = max(max_time, time)

                    ram = float(line.split("\t")[2])
                    ram_peak = max(ram_peak, ram)
                times[int(re.search("\d+", file).group(0))-1] = max_time - min_time
                ram_peaks[int(re.search("\d+", file).group(0))-1] = ram_peak

    max_n_codons_extrapolate = max_n_codons * 2
    max_n_codons_extrapolate = max_n_codons

    if bar:
        scale_time_by = 20000
        print("times =", times)
        times = times * scale_time_by
        print("ram_peaks =", ram_peaks)
        barWidth = 0.25
        fig = plt.subplots(figsize =(12, 8))


        br1_xpositions = np.arange(max_n_codons)
        br2_xpositions = [x + barWidth for x in br1_xpositions]

        plt.bar(br1_xpositions, times, color ='r', width = barWidth,
            edgecolor ='grey', label ='times in sec')
        plt.bar(br2_xpositions, ram_peaks, color ='b', width = barWidth,
            edgecolor ='grey', label ='ram_peaks in mb')

        plt.xticks([r + barWidth/2 for r in range(max_n_codons)],
                    np.arange(max_n_codons) + 1)

        def addlabels(x,y, scale_y = 1):
            for i in range(len(x)):
                plt.text(x[i],y[i],round(y[i] * scale_y))

        addlabels(br1_xpositions, times, 1/scale_time_by)
        addlabels(br2_xpositions, ram_peaks, 1/1024)
    else:
        fig, ax1 = plt.subplots()
        # plt.plot(, times, "bo-")
        # plt.plot(np.arange(max_n_codons), ram_peaks, "rx-")
        x = np.arange(1,max_n_codons +1)
        degree = 4
        # should coefs all be positive? for a runtime s
        coef_times = np.polyfit(x,times, degree) # coef for x^degree is coef_times[0]
        coef_ram_peaks = np.polyfit(x,ram_peaks/1024, degree)
        print("coef_times:", coef_times)
        print("coef_ram_peaks:", coef_ram_peaks)

        color = 'tab:red'
        ax1.set_xlabel('ncodons')
        ax1.set_ylabel('time in sec', color=color)
        ax1.plot(x, times, color=color)

        x_extrapolate = np.arange(1,max_n_codons_extrapolate+1)
        y = [np.polyval(coef_times,x) for x in x_extrapolate]
        ax1.plot(x_extrapolate, y, color = "tab:orange")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('ram_peak in mb', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, ram_peaks/1024, color=color)
        y = [np.polyval(coef_ram_peaks ,x) for x in x_extrapolate]
        ax2.plot(x_extrapolate, y, color = "tab:green")
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped



    plt.plot()
    plt.savefig("bench.png")

if __name__ == "__main__":
    plot_time_and_ram("bench")
########################################################################
########################################################################
########################################################################
def state_id_to_description(id, nCodons):
    states = re.split("\s+", "ig5' stA stT stG")
    states += ["c_" + str(i) + "," + str(j) for i in range(nCodons) for j in range(3)]
    states += re.split("\s+", "stop1 stop2 stop3 ig3'")
    states += ["i_" + str(i) + "," + str(j) for i in range(nCodons+1) for j in range(3)]
    states += ["ter1", "ter2"]
    return states[id]

def description_to_state_id(des, nCodons):
    states = re.split("\s+", "ig5' stA stT stG")
    states += ["c_" + str(i) + "," + str(j) for i in range(nCodons) for j in range(3)]
    states += re.split("\s+", "stop1 stop2 stop3 ig3'")
    states += ["i_" + str(i) + "," + str(j) for i in range(nCodons+1) for j in range(3)]
    states += ["ter1", "ter2"]
    try:
        return states.index(des)
    except:
        return -1

def higher_order_emission_to_id(emission, alphabet_size, order):
    # todo: emission 4,4,4 = I,I,I is not used, i might give this id to X
    # also 4,1,4 is not used
    if emission == "X" or emission ==  alphabet_size +1 or emission == [alphabet_size+1]:
        return (alphabet_size + 1)**(order + 1)
    #                                 initial symbol
    return sum([base*(alphabet_size + 1)**(len(emission) - i -1) for i, base in enumerate(emission)])

def id_to_higher_order_emission(id, alphabet_size, order):
    emission = []
    if id == (alphabet_size + 1)**(order + 1):
        return [alphabet_size +1]
    for i in range(order,0,-1):
        fits = int(id/((alphabet_size+1)**i))
        if fits < 1:
            emission += [0]
        else:
            id -= fits*((alphabet_size+1)**i)
            emission += [int(fits)]
    emission += [int(id)]
    return emission
########################################################################
########################################################################
########################################################################
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

        for i in sorted(list(log)):
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

# if __name__ == "__main__":
#     transform_verbose_txt_to_csv("verbose/1codons.txt",1)

########################################################################
########################################################################
########################################################################
def append_time_stamp_to_file(time, description, path):
    with open(path, "a") as file:
        file.write(f"{time}\t{description}\n")
def append_time_ram_stamp_to_file(start, description, path):
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

########################################################################
########################################################################
########################################################################
def run(command):
    import os
    os.system(f"echo '\033[96mrunning -> {command} \033[00m'")
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
########################################################################
########################################################################
########################################################################
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
########################################################################
########################################################################
########################################################################
def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

########################################################################
########################################################################
########################################################################

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
            alpha[q,i] = b[q,y[i]] * sum([a[q_,q] * alpha[q_,i-1]/z[i-1] for q_ in range(num_states)])
        z[i] = sum([alpha[q_,i] for q_ in range(num_states)])
    #P(Y=y)
    return alpha, z

def forward_felix_version_ordertransformedinput(a,b,y, a0 = []):
    # y is asumed to be one hot
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

########################################################################
########################################################################
########################################################################

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

########################################################################
########################################################################
########################################################################
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


########################################################################
########################################################################
########################################################################

# a = np.array([0.1,  0.2,   0.3,  0.2,  0.2,\
#               0.2,  0.2,   0.2,  0.2,  0.2,\
#               0.2,  0.15,  0.15, 0.3 , 0.2,\
#               0.3,  0.2,   0.4,  0.0,  0.1,\
#               0,    0.2,   0.5,  0.3,  0.0], dtype = np.float32).reshape((5,5))
#
# b = np.array([0.1,  0.2,   0.3,  0.4 ,\
#               0.2,  0.15,  0.15, 0.5 ,\
#               0.3,  0.2,   0.5,  0   ,\
#               0,    0.2,   0.5,  0.3 ,\
#               0.25, 0.25, 0.25,  0.25], dtype = np.float32).reshape((5,4))
#
# a = np.array([[.85,.15], [.1,.9]])
# b = np.array([[.1,.35,.35,.2], [.4,.1,.1,.4]])
#
# a0 = np.array([.3,.7])
#
# seqs = list(seqs.numpy())
# np.random.shuffle(seqs)
# y = seqs[0]
# y = y[:8]
# print(y)
# alpha, p = forward(a, b, np.argmax(y, axis = -1), a0)
# fullprint(np.transpose(alpha))
# print(p)
#
# alpha, p = forward_log_version(a, b, np.argmax(y, axis = -1), a0)
# fullprint(np.exp(np.transpose(alpha)))
# print(np.exp(p))
#
# alpha, z = forward_felix_version(a, b, np.argmax(y, axis = -1), a0)
# prod_z_up_to_i = 1
# print("z =", z)
# for i, i_row in enumerate(np.transpose(alpha)):
#     print("unscaled = ", i_row)
#     print("scaled = ", i_row * prod_z_up_to_i)
#     print("p up to now =", sum([(i_row * prod_z_up_to_i)[q] for q in range(len(a))]))
#
#     # update accumulating z for next interation of for loop
#     prod_z_up_to_i = prod_z_up_to_i * z[i]
# print("prod_z_up_to_i =", prod_z_up_to_i)
# # print("additional entry in z: p =", np.exp(sum([np.log(z[j]) for j in range(len(y))])+np.log(.843)))
# # print("p as sum of ln z =", np.exp(sum([np.log(z[j]) for j in range(len(y))])))
# # z[0] -= 0.0435
# # print("p as sum of ln z =", np.exp(sum([np.log(z[j]) for j in range(len(y))])))
# print("brute force P(Y=y) = ", brute_force_P_of_Y(a,b,np.argmax(y, axis = -1), a0))
