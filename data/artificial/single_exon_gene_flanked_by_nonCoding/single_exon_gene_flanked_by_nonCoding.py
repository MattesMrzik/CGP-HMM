#!/usr/bin/env python3
import subprocess
import numpy as np

def run(command):

    with open("temporary_script_file.sh","w") as file:
        file.write("#!/bin/bash\n")
        file.write(command)

    def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
    prRed("running " + command)

    subprocess.Popen("chmod +x temporary_script_file.sh".split(" ")).wait()
    subprocess.Popen("./temporary_script_file.sh").wait()
    subprocess.Popen("rm temporary_script_file.sh".split(" ")).wait()


with open("mytree.nwk","w") as file:
    # file.write("((human:0.1,chimpanse:0.1)Clade1:0.2,(mouse:0.2,rat:0.2)Clade2:0.1):0.1;\n")
    file.write("((human:0.1,chimpanse:0.1):0.2,(rat:0.2,mouse:0.2):0.1):0.1;\n")

def tree():
    id = [0]
    scale = .1
    min_evo_dist_to_child = 0.01
    def tree_rec(id, depth):
        if (np.random.rand() < .2 and depth > 2) or depth > 5:
            id[0] += 1
            return "id"+str(id[0]) + ":" + str(round(np.random.rand()*scale+min_evo_dist_to_child,2))
        else:
            return "(" + tree_rec(id, depth+1) + "," + tree_rec(id, depth+1) + "):" + str(round(np.random.rand()*scale+min_evo_dist_to_child,2))
    return tree_rec(id,0)

# write random newik tree to file
with open("mytree.nwk","w") as file:
    t = tree()
    print(t)
    file.write(t)

run("../seq-gen -m GTR -l 6 mytree.nwk -of > seq_gen.out")
run("cat mytree.nwk")
run("cat seq_gen.out")


with open("seq_gen.out","r") as infile:
    with open("seq_gen.out.with_utr.fasta","w") as outfile:
        for line in infile:
            if line[0] == ">":
                outfile.write(line)
            else:
                random_start_pos = np.random.randint(15)
                # utr5 = "".join(np.random.choice(["A","C","G","T"], np.random.randint(14,15)))
                # utr3 = "".join(np.random.choice(["A","C","G","T"], np.random.randint(14,15)))

                utr5 = "".join(np.random.choice(["A","C","G","T"], random_start_pos))
                utr3 = "".join(np.random.choice(["A","C","G","T"], 15- random_start_pos))
                outfile.write(utr5 + "ATG" + line.strip() + np.random.choice(["TAG", "TAA", "TGA"]) + utr3 + "\n")

run("cat seq_gen.out.with_utr.fasta")
run("../muscle3.8.31_i86linux64 -in seq_gen.out.with_utr.fasta -out seq_gen.out.with_utr.fasta.alignment.fa")
run("cat seq_gen.out.with_utr.fasta.alignment.fa")
run("sed g\;n seq_gen.out.with_utr.fasta.alignment.fa | cut -c 18|sort | uniq -c")
