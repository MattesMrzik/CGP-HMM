#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from Bio import SeqIO
from CgpHmmLayer import CgpHmmLayer
from CgpHmmCell import CgpHmmCell

import matplotlib.pyplot as plt


# import tensorflow_datasets as tfds
#
# tfds.disable_progress_bar()

# dataset, info = tfds.load('imdb_reviews', with_info=True,
#                           as_supervised=True)
# train_dataset, test_dataset = dataset['train'], dataset['test']
# print(train_dataset)

def prRed(skk): print("Training\033[92m {}\033[00m" .format(skk))
np.set_printoptions(linewidth=400)

def make_model():
    alphabet_size = 4

    prRed("22sequences = tf.keras.Input(shape = (None, alphabet_size), name = \"sequences\")")
    sequences = tf.keras.Input(shape = (50, alphabet_size), name = "sequences")
    print("24sequences.shape = ", sequences.shape)
    # another None added automatically for yet unkown batch_size
    # todo what if sequences have differing lenghts, then the 24 cant stay, padding?

    prRed("28cgp_hmm_layer = CgpHmmLayer()")
    cgp_hmm_layer = CgpHmmLayer()

    prRed("31loglik = cgp_hmm_layer(sequences)")
    loglik = cgp_hmm_layer(sequences)
    prRed("33loglik")
    print(loglik)

    prRed("36[tf.keras.layers.Lambda(lambda x:x, name = \"loglik\")(loglik)]")
    print([tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)])

    prRed("39model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = \"loglik\")(loglik)])")
    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)]) #  the output of the model is the value that is computed by a final layer that picks the loglike of the [alpha, loglik, count]

    prRed("42return model, cgp_hmm_layer")
    return model, cgp_hmm_layer


def make_dataset():

    # with open("identical_seqs_with_repeates.out","w") as file:
    #     for i in range(10):
    #         file.write(">id" + str(i) + "\n")
    #         file.write("ACGT"*3 + "\n")

    with open("intron_AT_exon_GC.out","w") as file:
        for i in range(1000):
            file.write(">id" + str(i) + "\n")
            alphabet = ["A","C","G","T"]
            a = np.array([[.85,.15], [.1,.9]])
            b = np.array([[.1,.35,.35,.2], [.4,.1,.1,.4]])

            current_state = 0
            seq_length = 50
            seq = ""
            for i in range(seq_length):
                seq += np.random.choice(alphabet, size = 1, p = b[current_state])[0]
                current_state = np.random.choice([0,1], size = 1, p = a[current_state])[0]
            file.write(seq + "\n")

    seqs = []
    prRed("make_dataset")
    alphabet = ["A","C","G","T"]
    AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])
    max_genomes = float('inf')
    count_genomes = 0
    with open("intron_AT_exon_GC.out","r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq

            print(seq, ", lenght: ", len(seq))

            seq = list(map(lambda x: AA_to_id[x], seq))
            seqs.append(seq)
            count_genomes += 1
            if count_genomes == max_genomes:
                break
            # prRed("[one_hot]")
            # print(one_hot)
            # seqs = np.array(np.array(one_hot)) if len(seq) == 0 else np.append(seqs, one_hot, axis = 0)
            # print(seqs)

    one_hot = tf.one_hot(seqs, len(alphabet))
    # prRed("one_hot")
    # tf.print(one_hot, summarize=100)
    prRed("ds = tf.data.Dataset.from_tensor_slices(one_hot)")
    ds = tf.data.Dataset.from_tensors(one_hot).repeat()
    print("ds =", ds)
    return ds, one_hot

def fit_model():
    # make model
    prRed("model, cgp_hmm_layer = make_model()")
    model, cgp_hmm_layer = make_model()#
    # compile model
    learning_rate = .1
    optimizer = tf.optimizers.Adam(learning_rate)
    prRed("model.compile(optimizer = optimizer)")
    model.compile(optimizer = optimizer)


    # _, seqs = make_dataset()# first return value is data_set
    # model(seqs)
    # prRed("90manual call done")

    # model.fit
    prRed("data_set = make_dataset()[0]")
    data_set = make_dataset()[0] # [1] is data tensor
    prRed("96data_set")
    print(data_set)
    prRed("history = model.fit(data_set, epochs=2, steps_per_epoch=2)")

    class my_callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            print("model.weights")
            print("A =", tf.nn.softmax(model.get_weights()[0]))

    # callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: print("A =", tf.nn.softmax(model.get_weights()[0])))]
    callbacks = [my_callback()]
    # callbacks = []

    # return
    history = model.fit(data_set, epochs=5, steps_per_epoch=100, callbacks = callbacks) # with callbacks it is way slower
    return model, history

model, history = fit_model()

plt.plot(history.history['loss'])
plt.show()
prRed("weights")

print("A =", tf.math.round(tf.nn.softmax(model.get_weights()[0])*100)/100)
print("B =", tf.math.round(tf.nn.softmax(model.get_weights()[1])*100)/100)
