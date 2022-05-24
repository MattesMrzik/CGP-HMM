#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from Bio import SeqIO
from CgpHmmLayer import CgpHmmLayer
from CgpHmmCell import CgpHmmCell

# import tensorflow_datasets as tfds
#
# tfds.disable_progress_bar()

# dataset, info = tfds.load('imdb_reviews', with_info=True,
#                           as_supervised=True)
# train_dataset, test_dataset = dataset['train'], dataset['test']
# print(train_dataset)

def prRed(skk): print("Training\033[91m {}\033[00m" .format(skk))

def make_model():
    alphabet_size = 4

    prRed("sequences = tf.keras.Input(shape = (None, alphabet_size), name = \"sequences\")")
    sequences = tf.keras.Input(shape = (24, alphabet_size), name = "sequences")
    print("sequences.shape = ", sequences.shape)
    # another None added automatically for yet unkown batch_size
    # todo what if sequences have differing lenghts, then the 24 cant stay, padding?

    prRed("cgp_hmm_layer = CgpHmmLayer()")
    cgp_hmm_layer = CgpHmmLayer()

    prRed("likelihood = cgp_hmm_layer(sequences)")
    likelihood = cgp_hmm_layer(sequences)

    prRed("[tf.keras.layers.Lambda(lambda x:x, name = \"likelihood\")(likelihood)]")
    print([tf.keras.layers.Lambda(lambda x:x, name = "likelihood")(likelihood)])

    prRed("model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = \"likelihood\")(likelihood)])")
    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "likelihood")(likelihood)])

    prRed("return model, cgp_hmm_layer")
    return model, cgp_hmm_layer


def make_dataset():
    seqs = []
    prRed("make_dataset")
    alphabet = ["A","C","G","T"]
    AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])
    max_genomes = 7
    count_genomes = 0
    with open("seq-gen.out","r") as handle:
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
    prRed("one_hot")
    # tf.print(one_hot, summarize=100)
    prRed("ds = tf.data.Dataset.from_tensor_slices(one_hot)")
    ds = tf.data.Dataset.from_tensors(one_hot)
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


    _, seqs = make_dataset()
    model(seqs)
    prRed("manual call done")

    # model.fit
    prRed("history = model.fit(make_dataset(), epochs=2, steps_per_epoch=2)")

    data_set = make_dataset()[0] # [1] is data tensor
    prRed("data_set")
    print(data_set)
    history = model.fit(data_set, epochs=2, steps_per_epoch=2)

    return model, history

fit_model()
