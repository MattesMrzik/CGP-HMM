#!/usr/bin/env python3
import numpy as np
from Bio import SeqIO
from ReadData import read_data_one_hot

np.set_printoptions(linewidth=200)

########################################################################
########################################################################
########################################################################
def generate_state_emission_seqs(a,b,n,l, a0 = []):

    state_space_size = len(a)
    emission_state_size = len(b[0])

    list_of_state_space_size = list(range(state_space_size))
    list_of_emission_state_size = list(range(emission_state_size))


    states = np.zeros((n,l,state_space_size))
    emissions = np.zeros((n,l,emission_state_size))

    def rand_with_p(list, dist):
        return np.random.choice(list, size = 1, p = dist)[0]

    for i in range(n):
        states[i,0,0 if len(a0) == 0 else rand_with_p(list_of_state_space_size, a0)] = 1
        emissions[i,0, rand_with_p(list_of_emission_state_size, b[np.argmax(states[0,0,:])])] = 1
        for j in range(l):
            states[i,j, rand_with_p(list_of_state_space_size, a[np.argmax(states[i,j-1])])] = 1
            emissions[i,j, rand_with_p(emission_state_size, b[np.argmax(states[i,j-1])])] = 1
        # emssions.write(seq + "\n")
        # states.write(">id" + str(i) + "\n")
        # states.write(state_seq + "\n")
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

# farward = P_theta(Y)
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
    p = b[x[0], y[0]] if len(a0) == 0 else a0[x[0]] * b[x[0], y[0]]
    for i in range(1,len(y)):
        p *= a[x[i-1], x[i]]
        p *= b[x[i], y[i]]
        return p

def P_of_X_Y_log_version(a,b,x,y, a0 =[]):
    p = np.log(b[0, y[0]]) if len(a0) == 0 else np.log(a0[x[0]] * b[x[0], y[0]])
    for i in range(1,len(y)):
        p += np.log(a[x[i-1], x[i]])
        p += np.log(b[x[i], y[i]])
    return p

########################################################################
########################################################################
########################################################################
# argmax_x: P_theta(x|y)
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

a = np.array([0.1,  0.2,   0.3,  0.2,  0.2,\
              0.2,  0.2,   0.2,  0.2,  0.2,\
              0.2,  0.15,  0.15, 0.3 , 0.2,\
              0.3,  0.2,   0.4,  0.0,  0.1,\
              0,    0.2,   0.5,  0.3,  0.0], dtype = np.float32).reshape((5,5))

b = np.array([0.1,  0.2,   0.3,  0.4 ,\
              0.2,  0.15,  0.15, 0.5 ,\
              0.3,  0.2,   0.5,  0   ,\
              0,    0.2,   0.5,  0.3 ,\
              0.25, 0.25, 0.25,  0.25], dtype = np.float32).reshape((5,4))

a = np.array([[.85,.15], [.1,.9]])
b = np.array([[.1,.35,.35,.2], [.4,.1,.1,.4]])

a0 = np.array([.3,.7])

seqs = read_data_one_hot("../rest/seq-gen.out")
seqs = list(seqs.numpy())
np.random.shuffle(seqs)
y = seqs[0]
y = y[:8]
print(y)
alpha, p = forward(a, b, np.argmax(y, axis = -1), a0)
fullprint(np.transpose(alpha))
print(p)

alpha, p = forward_log_version(a, b, np.argmax(y, axis = -1), a0)
fullprint(np.exp(np.transpose(alpha)))
print(np.exp(p))

alpha, z = forward_felix_version(a, b, np.argmax(y, axis = -1), a0)
prod_z_up_to_i = 1
print("z =", z)
for i, i_row in enumerate(np.transpose(alpha)):
    print("unscaled = ", i_row)
    print("scaled = ", i_row * prod_z_up_to_i)
    print("p up to now =", sum([(i_row * prod_z_up_to_i)[q] for q in range(len(a))]))

    # update accumulating z for next interation of for loop
    prod_z_up_to_i = prod_z_up_to_i * z[i]
print("prod_z_up_to_i =", prod_z_up_to_i)
# print("additional entry in z: p =", np.exp(sum([np.log(z[j]) for j in range(len(y))])+np.log(.843)))
# print("p as sum of ln z =", np.exp(sum([np.log(z[j]) for j in range(len(y))])))
# z[0] -= 0.0435
# print("p as sum of ln z =", np.exp(sum([np.log(z[j]) for j in range(len(y))])))
print("brute force P(Y=y) = ", brute_force_P_of_Y(a,b,np.argmax(y, axis = -1), a0))
