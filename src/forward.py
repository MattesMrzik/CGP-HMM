#!/usr/bin/env python3
import numpy as np
from Bio import SeqIO

np.set_printoptions(linewidth=200)

a = np.array([0.1,  0.2,   0.3,  0.2,  0.2,\
                         0.2,  0.2,   0.2,  0.2,  0.2,\
                         0.2,  0.15,  0.15, 0.3 , 0.2,\
                         0.3,  0.2,   0.4,  0.0,  0.1,\
                         0,    0.2,   0.5,  0.3,  0.0], dtype = np.float32).reshape((5,5))

b =np.array([0.1,  0.2,   0.3,  0.4 ,\
                         0.2,  0.15,  0.15, 0.5 ,\
                         0.3,  0.2,   0.5,  0   ,\
                         0,    0.2,   0.5,  0.3 ,\
                         0.25, 0.25, 0.25,  0.25], dtype = np.float32).reshape((5,4))


alphabet = ["A","C","G","T"]
AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])

num_states = 5
seqs = []
with open("seq-gen.out","r") as handle:
    for record in SeqIO.parse(handle,"fasta"):
        seq = record.seq
        seq = list(map(lambda x: AA_to_id[x], seq))
        seqs.append(seq)

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


y = seqs[0]
print("y =")
fullprint(np.array(y))
alpha = np.zeros((num_states,len(y)))
alpha[0,0] = b[0,y[0]] # one must start in the first state

for i in range(1,len(y)):
    for q in range(num_states):
        alpha[q,i]=b[q,y[i]]*sum([a[q_,q]*alpha[q_,i-1] for q_ in range(num_states)])
print("alpha =")
fullprint(np.transpose(alpha))

#P(Y=y)
p = sum([alpha[q,len(y)-1] for q in range(len(alphabet))])
print("P(Y=y) =", p)
