#!/usr/bin/env python3

from Utility import run
import numpy as np
import re
from Utility import state_id_to_description
from Utility import id_to_higher_order_emission

import argparse

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', '--nCodons', required=True,
                    help='number of codons')
parser.add_argument('-o', '--order', required=True,
                    help='order of the emission model')

parser.add_argument('-t', action='store_true', help ="if passed, then order_transformed_input is used")

args = parser.parse_args()

try:
    args.order = int(args.order)
except:
    print("args.order must be int")

nCodons = int(args.nCodons)

n_most_likely_emissions = 3
id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}
with open(f"output/{nCodons}codons/graph.{nCodons}codons.gv", "w") as graph:

    most_likely = {}
    with open(f"output/{nCodons}codons/B.{nCodons}codons.txt", "r") as b:
        for line in b:
            line = re.sub("[(|)| ]","", line.strip())
            line = re.split(",|;", line)
            state = state_id_to_description(int(line[0]), nCodons)
            prob = float(line[-1])
            if args.t:
                emissions_tuple = id_to_higher_order_emission(int(line[1]), len(id_to_base)-2, args.order)
                emissions_tuple = ("".join(list(map(lambda x: id_to_base[int(x)], emissions_tuple))), np.round(prob,4))
            else:
                emissions_tuple = ("".join(list(map(lambda x: id_to_base[int(x)], line[1:-1]))), np.round(prob,4))
            if not state in most_likely:
                most_likely[state] = [emissions_tuple]
            else:
                most_likely[state].append(emissions_tuple)

    for key in most_likely.keys():
        most_likely[key] = sorted(most_likely[key], key = lambda x: x[1], reverse = True)
        # print(most_likely[key])
    graph.write("DiGraph G{\nrankdir=LR;\n")
    # graph.write("nodesep=0.5; splines=polyline;")

    with open(f"output/{nCodons}codons/A.{nCodons}codons.txt","r") as a:
        for line in a:
            line = re.sub("[(|)| ]","", line.strip())
            line = re.split(",|;", line)
            i = state_id_to_description(int(line[0]), nCodons)
            j = state_id_to_description(int(line[1]), nCodons)
            if i == state_id_to_description(0,nCodons):
                # graph.write("\""+ j +"\" [shape=record label=\"{{ " + j + "|" + "|".join([str(most_likely[j][k]) for k in range(n_most_likely_emissions)]) + "}}\"];\n")

                # or

                graph.write("\"" + j + "\"\n")
                graph.write("[\n")
                graph.write("\tshape = none\n")
                graph.write("\tlabel = <<table border=\"0\" cellspacing=\"0\"> \n")
                try:
                    color = {"c_":"teal", "i_": "crimson"}[j[0:2]]
                except:
                    color = "white"
                graph.write(f"\t\t<tr><td port=\"port1\" border=\"1\" bgcolor=\"{color}\">" + j + "</td></tr>\n")
                for k in range(n_most_likely_emissions):
                    graph.write(f"\t\t<tr><td port=\"port{k+2}\" border=\"1\">{most_likely[j][k]}</td></tr>\n" )
                graph.write("\t </table>>\n")
                graph.write("]\n")

            prob = float(line[2])
            if prob > 0:
                graph.write(f"\"{i}\" -> \"{j}\" [label = {np.round(prob, 4)} fontsize=\"{30*prob + 5}pt\"]\n")
    graph.write("}")
# run(f"cat graph.{nCodons}codons.gv")
run(f"dot -Tpng output/{nCodons}codons/graph.{nCodons}codons.gv -o output/{nCodons}codons/graph.{nCodons}codons.png")
