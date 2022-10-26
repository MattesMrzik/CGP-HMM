#!/usr/bin/env python3
from Utility import state_id_to_description

def write_to_file(matrix, path):
    from itertools import product
    import tensorflow as tf
    # try:
    with open(path, "w") as file:
        x = [list(range(dim_size)) for dim_size in tf.shape(matrix)]
        for index in product(*x):
            file.write(str(index))
            file.write(";")
            file.write(str(matrix[index].numpy()))
            file.write("\n")
    # except:
    #     print("could not write to file:", path)
    #     quit(1)

def write_order_transformed_B_to_csv(b, path, order, nCodons, alphabet = ["A","C","G","T"]):
    from itertools import product
    import tensorflow as tf
    b = tf.transpose(b)
    print(tf.shape(b))
    with open(path, "w") as file:
        # write emissions
        for emission in [""] + list(product(alphabet + ["I"], repeat = order +1)) + ["X"]:
            file.write("".join(emission))
            file.write("\t")
        file.write("\n")

        for state_id in range(tf.shape(b)[0]):
            file.write(str(state_id_to_description(state_id, nCodons)))
            file.write("\t")
            for prob in b[state_id]:
                file.write(str(prob.numpy()))
                file.write("\t")
            file.write("\n")
