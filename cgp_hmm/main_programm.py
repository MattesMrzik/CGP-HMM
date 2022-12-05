#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='description')

# base algo
parser.add_argument('-c', '--nCodons', help='number of codons')
parser.add_argument('-t', '--type', help='type of cell.call():  0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel')
parser.add_argument('-p', '--path', help='path to src')

# fine tune algo
parser.add_argument('--use_weights_for_consts', action='store_true', help ="use weights for transitions that become 1 after softmax")
parser.add_argument('--weaken_softmax', action='store_true', help ="weaken_softmax such that after softmax the are no near zero or zero values")
parser.add_argument('-d', '--dytpe64', action='store_true', help='using dytpe tf.float64')
parser.add_argument('--clip_gradient_by_value', help ="clip_gradient_by_values", type = float)
parser.add_argument('--learning_rate', help ="learning_rate", type = float)
parser.add_argument('--no_learning', help ="learning_rate is set to 0", action='store_true', )
parser.add_argument('-l',help = 'lenght of onput seqs when using MSAgen')

# hardware
parser.add_argument('--split_gpu', action='store_true', help ="split gpu into 2 logical devices")

# verbose
parser.add_argument('-v', '--verbose', nargs = "?", const = "2", help ="verbose E,R, alpha, A, B to file, pass 1 for shapes, 2 for shapes and values")
parser.add_argument('-o', '--only_keep_verbose_of_last_batch', action='store_true', help ="only_keep_verbose_of_last_batch")
parser.add_argument('-s', '--verbose_to_stdout', action='store_true', help ="verbose to stdout instead of to file")
parser.add_argument('--cpu_gpu', action='store_true', help ="print whether gpu or cpu is used")
parser.add_argument('--batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs', action='store_true', help ="batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs")
parser.add_argument('--get_gradient_of_first_batch', action='store_true', help ="get_gradient_of_first_batch")
parser.add_argument('--get_gradient_for_current_txt', action='store_true', help ="get_gradient_for_current_txt, previous run wrote IAB and inputbath to file -> get respective gradient")
parser.add_argument('--get_gradient_in_layer', action='store_true', help ="get_gradient_for current values directly in the call of layer, but 'Gradient for SparseDenseCwiseAdd is not implemented.'")
parser.add_argument('--get_gradient_from_saved_model_weights', action='store_true', help ="get_gradient_from_saved_model_weights, they are saved when passing --most_recent_weights_and_inputs_to_file")



# debugging
parser.add_argument('-b', action='store_true', help ="exit after first batch, you may use this when verbose is True in cell.call()")
parser.add_argument('-n', action='store_true', help ="exit_after_loglik_is_nan, you may use this when verbose is True in cell.call()")
parser.add_argument('--dont_generate_new_seqs', action='store_true', help ="dont_generate_new_seqs, but use the ones that were created before")

args = parser.parse_args()

config = {}

config["nCodons"] = int(args.nCodons) if args.nCodons else 1
config["order"] = 2
config["order_transformed_input"] = True
config["call_type"] = int(args.type) if args.type else 3 # 0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel

config["alphabet_size"] = 4
config["src_path"] = "." if not args.path else args.path
config["fasta_path"] = f"{config['src_path']}/output/{config['nCodons']}codons/out.seqs.{config['nCodons']}codons.fa"
config["bench_path"] = f"{config['src_path']}/bench/{config['nCodons']}codons/{config['call_type']}_{config['order_transformed_input']}orderTransformedInput.log"
config["exit_after_first_batch"] = args.b
config["exit_after_loglik_is_nan"] = args.n
config["verbose"] = int(args.verbose) if args.verbose else 0
config["print_to_file"] = not args.verbose_to_stdout
config["dtype"] = "tf.float64" if args.dytpe64 else "tf.float32"
config["use_weights_for_consts"] = args.use_weights_for_consts
config["only_keep_verbose_of_last_batch"] = args.only_keep_verbose_of_last_batch
config["weaken_softmax"] = args.weaken_softmax
config["get_gradient_of_first_batch"] = args.get_gradient_of_first_batch
if args.clip_gradient_by_value:
    config["clip_gradient_by_value"] = args.clip_gradient_by_value
config["learning_rate"] = args.learning_rate if args.learning_rate else 0.1
if args.no_learning:
    config["learning_rate"] = 0
config["dont_generate_new_seqs"] = args.dont_generate_new_seqs
config["batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs"] = args.batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs
config["get_gradient_for_current_txt"] = args.get_gradient_for_current_txt
config["get_gradient_in_layer"] = args.get_gradient_in_layer
config["get_gradient_from_saved_model_weights"] = args.get_gradient_from_saved_model_weights

from Utility import get_state_id_description_list
config["state_id_description_list"] = get_state_id_description_list(config["nCodons"])

from Utility import get_indices_for_weights_from_transition_kernel_higher_order
from Utility import get_indices_for_constants_from_transition_kernel_higher_order
from Utility import get_indices_for_weights_from_emission_kernel_higher_order
from Utility import get_indices_for_constants_from_emission_kernel_higher_order
from Utility import get_indices_from_initial_kernel
from Utility import run
import Utility

nCodons = config["nCodons"]

run(f"mkdir -p {config['src_path']}/output/{nCodons}codons/")
run(f"mkdir -p {config['src_path']}/verbose")
run(f"mkdir -p {'/'.join(config['bench_path'].split('/')[:-1])}")
run(f"rm {config['src_path']}/{config['bench_path']}")
run(f"rm {config['src_path']}/verbose/{nCodons}codons.txt")

Utility.get_indices_for_config(config)

Utility.print_config(config)

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

config["dtype"] = tf.float64 if args.dytpe64 else tf.float32

if config["dtype"] == tf.float64:
    policy = tf.keras.mixed_precision.Policy("float64")
    tf.keras.mixed_precision.set_global_policy(policy)

if args.cpu_gpu:
    tf.debugging.set_log_device_placement(True) # shows whether cpu or gpu is used

num_physical_gpus = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_physical_gpus, "tf.config.list_physical_devices")

if num_physical_gpus and args.split_gpu:
    phisical_gpus = tf.config.experimental.list_physical_devices("GPU")
    print(phisical_gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    tf.config.experimental.set_virtual_device_configuration(
        phisical_gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
    )
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("logical_gpus =", logical_gpus, "tf.config.experimental.list_logical_devices")
    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

    print("Using", num_gpu, "GPUs. device_lib.list_local_devices()")

    print("printing local devices")
    for i, x in  enumerate(device_lib.list_local_devices()):
        print(i, x.name)


if not args.dont_generate_new_seqs:
    run(f"python3 {config['src_path']}/useMSAgen.py -c {nCodons} {'-l' + args.l if args.l else ''}")

model, history = fit_model(config)
print("done fit_model()")
# model.save("my_saved_model")

with open(f"{config['src_path']}/output/{nCodons}codons/loss.log", "w") as file:
    for loss in history.history['loss']:
        file.write(str(loss))
        file.write("\n")

plt.plot(history.history['loss'])
plt.savefig(f"{config['src_path']}/progress.png")

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
if not config["call_type"] == 4:
    WriteData.write_to_file(cell.A_dense, f"{config['src_path']}/output/{nCodons}codons/A.{nCodons}codons.txt")
    WriteData.write_to_file(tf.transpose(cell.B_dense), f"{config['src_path']}/output/{nCodons}codons/B.{nCodons}codons.txt")
    WriteData.write_order_transformed_B_to_csv(cell.B_dense, f"{config['src_path']}/output/{nCodons}codons/B.{nCodons}codons.csv", config["order"], nCodons)

    WriteData.write_to_file(cell.I_dense, f"{config['src_path']}/output/{nCodons}codons/I.{nCodons}codons.txt")

    # running Viterbi
    run(f"{config['src_path']}/Viterbi " + config["fasta_path"] + " " + str(nCodons))

    stats = {"start_not_found" : 0,\
             "start_too_early" : 0,\
             "start_correct" : 0,\
             "start_too_late" : 0,\
             "stop_not_found" : 0,\
             "stop_too_early" : 0,\
             "stop_correct" : 0,\
             "stop_too_late" : 0}

    with open(f"{config['src_path']}/output/{nCodons}codons/viterbi.{nCodons}codons.csv", "r") as viterbi_file:
        with open(f"{config['src_path']}/output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt", "r") as start_stop_file:
            for v_line in viterbi_file:
                try:
                    ss_line = start_stop_file.readline()
                except:
                    print("ran out of line in :" + f"out.start_stop_pos.{nCodons}codons.txt")
                    quit(1)
                if ss_line[:3] == "seq" or len(ss_line) <= 1:
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

    with open(f"{config['src_path']}/output/{nCodons}codons/statistics.txt", "w") as file:
        for key, value in stats.items():
            file.write(key + "\t" + str(value/nSeqs) + "\n")

if config["nCodons"] < 10:
    run(f"python3 {config['src_path']}/Visualize.py -c {nCodons} -o {config['order']} {'-t' if config['order_transformed_input'] else ''}")
