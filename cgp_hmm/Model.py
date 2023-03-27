#!/usr/bin/env python3
import json
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
class Model(ABC):

    def __init__(self, config):
        self.config = config

    # kernel sizes
    @abstractmethod
    def I_kernel_size(self):
        pass

    def A_kernel_size(self):
        pass

    def B_kernel_size(self):
        pass

    # matrices
    @abstractmethod
    def I(self, weights):
        pass

    @abstractmethod
    def A(self, weights):
        pass

    @abstractmethod
    def B(self, weights):
        pass



    @abstractmethod
    def get_number_of_states():
        pass

    @abstractmethod
    def get_number_of_emissions():
        pass

    @abstractmethod
    def state_id_to_str():
        pass

    @abstractmethod
    def str_to_state_id():
        pass

    @abstractmethod
    def emission_id_to_str():
        pass

    @abstractmethod
    def str_to_emission_id():
        pass

    def write_model(self):
        pass
    def read_model(self):
        pass
    def find_indices_of_zeros():
        pass

    def fasta_true_state_seq_and_optional_viterbi_guess_alignment(self, fasta_path, viterbi_path = None):

        # assumes viterbi only contains prediction for human

        from Bio import SeqIO, AlignIO
        from Bio.Align import MultipleSeqAlignment
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        import re

        try:
            fasta_data = SeqIO.parse(fasta_path, "fasta")
            for record in fasta_data:
                if re.search("Homo_sapiens", record.id):
                    human_fasta = record
                    # if nothing is found this will call except block
                    human_fasta.id
        except:
            print("seqIO could not parse", file)
            return

        coords = json.loads(re.search("({.*})", human_fasta.description).group(1))

        l = []
        if viterbi_path != None:
            try:
                file = open(viterbi_path)
            except:
                print("could not open", file)
                return
            try:
                json_data = json.load(file)
            except:
                print("json could not parse", file)
                return

            if type(json_data[0]) is list: #[[0,1,2],[0,0,1],[1,2,3,4,5]]
                description_seq = []
                for seq_id, seq in enumerate(json_data):
                    for nth_state, state in enumerate(seq):
                        description = self.state_id_to_str(state)
                        description_seq.append((state,description))
                    l.append(description_seq)
            else: # [0,0,0,01,2,3,4,4,4,4]
                for nth_state, state in enumerate(json_data):
                    description = self.state_id_to_str(state)
                    l.append((state,description))


################################################################################
        viterbi_as_fasta = ""
        if fasta_path == None:
            viterbi_as_fasta = " " * len(human_fasta.seq)
        else:
            for state_id, description in l[0]:
                if description == "left_intron":
                    viterbi_as_fasta += "l"
                elif description == "right_intron":
                    viterbi_as_fasta += "r"
                elif description == "A":
                    viterbi_as_fasta += "A"
                elif description == "AG":
                    viterbi_as_fasta += "G"
                elif description == "G":
                    viterbi_as_fasta += "G"
                elif description == "GT":
                    viterbi_as_fasta += "T"
                else:
                    viterbi_as_fasta += "-"

            # removing terminal
            viterbi_as_fasta = viterbi_as_fasta[:-1]
            assert l[0][-1][1] == "ter", "Model.py last not terminal"

        viterbi_record = SeqRecord(seq = Seq(viterbi_as_fasta), id = "viterbi_guess")
################################################################################
        on_reverse_strand = coords["exon_start_in_human_genome_cd_strand"] != coords["exon_start_in_human_genome_+_strand"]
        if not on_reverse_strand:
            true_seq = "l" * (coords["exon_start_in_human_genome_+_strand"] - coords["seq_start_in_genome_+_strand"])
            true_seq += "E" * (coords["exon_stop_in_human_genome_+_strand"] - coords["exon_start_in_human_genome_+_strand"])
            true_seq += "r" * (coords["seq_stop_in_genome_+_strand"] - coords["exon_stop_in_human_genome_+_strand"])
        else:
            true_seq = "l" * (coords["seq_start_in_genome_cd_strand"] - coords["exon_start_in_human_genome_cd_strand"])
            true_seq += "E" * (coords["exon_start_in_human_genome_cd_strand"] - coords["exon_stop_in_human_genome_cd_strand"])
            true_seq += "r" * (coords["exon_stop_in_human_genome_cd_strand"] - coords["seq_stop_in_genome_cd_strand"])
        true_seq_record = SeqRecord(seq = Seq(true_seq), id = "true_seq")
################################################################################
        len_of_line_in_clw = 50
        numerate_line = ""
        for i in range(len(viterbi_as_fasta)):
            i_line = i % len_of_line_in_clw
            if i_line % 10 == 0:
                numerate_line += "|"
            else:
                numerate_line += " "

        numerate_line_record =  SeqRecord(seq = Seq(numerate_line), id = "numerate_line")
################################################################################
        coords_fasta = ""
        for line_id in range(len(viterbi_as_fasta)//len_of_line_in_clw):
            in_fasta = line_id*len_of_line_in_clw
            if not on_reverse_strand:
                coords_line = f"in this fasta {in_fasta}, in genome {in_fasta + coords['seq_start_in_genome_+_strand']}"
            else:
                coords_line = f"in this fasta {in_fasta}, in genome {coords['seq_start_in_genome_cd_strand']- in_fasta}"
            coords_fasta += coords_line + " " * (len_of_line_in_clw - len(coords_line))

        last_line_len = len(viterbi_as_fasta) - len(coords_fasta)
        coords_fasta += " " * last_line_len

        coords_fasta_record = SeqRecord(seq = Seq(coords_fasta), id = "coords_fasta")

################################################################################
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record, viterbi_record]
        alignment = MultipleSeqAlignment(records)

        alignment_out_path = f"{os.path.dirname(viterbi_path)}/true_alignment.txt" if viterbi_path != None else f"{os.path.dirname(fasta_path)}/true_alignment.txt"
        with open(alignment_out_path, "w") as output_handle:
            AlignIO.write(alignment, output_handle, "clustal")
        print("wrote alignment to", alignment_out_path)
        
        return l
################################################################################
################################################################################
################################################################################
    def export_to_dot_and_png(self, A_weights, B_weights, out_path = "this is still hard coded"):
        # TODO: add I parameters???
        n_labels = self.number_of_emissions ** (self.config.order + 1)
        nCodons = self.config.nCodons

        A = self.A(A_weights) if self.A_is_dense else tf.sparse.to_dense(self.A(A_weights))
        B = self.B(B_weights) if self.B_is_dense else tf.sparse.to_dense(self.B(B_weights))

        B_reshaped = tf.reshape(B, shape = (-1, self.config.alphabet_size, self.number_of_states))
        B_argmax = np.argmax(B_reshaped, axis = 1)

        id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}
        with open(f"output/{nCodons}codons/graph.gv", "w") as graph:
            graph.write("DiGraph G{\nrankdir=LR;\n")
            # graph.write("nodesep=0.5; splines=polyline;")
            for from_state, row in enumerate(A):
                from_state_str = self.state_id_to_str(from_state)
                graph.write("\"" + from_state_str + "\"\n") #  this was to_state before

                graph.write("[\n")
                graph.write("\tshape = none\n")
                graph.write("\tlabel = <<table border=\"0\" cellspacing=\"0\"> \n")
                try:
                    color = {"c_":"teal", "i_": "crimson"}[from_state_str[0:2]]
                except:
                    color = "white"

                graph.write(f"\t\t<tr><td port=\"port1\" border=\"1\" bgcolor=\"{color}\">" + from_state_str + "</td></tr>\n")

                for k, most_likely_index in enumerate(B_argmax[:,from_state]):
                    emission_id = most_likely_index + k * self.config.alphabet_size
                    emission_str = self.emission_id_to_str(emission_id)
                    emi_prob = str(np.round(B[emission_id, from_state].numpy(),4))
                    graph.write(f"\t\t<tr><td port=\"port{k+2}\" border=\"1\">({emission_str + ' ' +emi_prob})</td></tr>\n" )
                graph.write("\t </table>>\n")
                graph.write("]\n")

                for to_state, prob in enumerate(row):
                    to_state_str = self.state_id_to_str(to_state)
                    if prob > 0:
                        prob = prob.numpy()
                        graph.write(f"\"{from_state_str}\" -> \"{to_state_str}\" [label = {str(np.round(prob, 4))[:6]} fontsize=\"{30*prob + 5}pt\"]\n")

            graph.write("}")
        # run(f"cat graph.{nCodons}codons.gv")
        from Utility import run
        run(f"dot -Tpng output/{nCodons}codons/graph.gv -o output/{nCodons}codons/graph.png")
