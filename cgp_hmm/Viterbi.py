#!/usr/bin/env python3
import os
import pandas as pd
import json
import re
import time
from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from helpers.convert_kernels_to_matrices import convert_kernel_files_to_matrices_files

def get_true_state_seqs_from_true_MSA(config):
    # calc true state seq from true MSA
    true_state_seqs = []
    msa_state_seq = ""
    with open(f"{config.current_run_dir}/trueMSA.txt","r") as msa:
        for j, line in enumerate(msa):
            true_state_seq = []
            if j == 0:
                msa_state_seq = line.strip()
                continue
            which_codon = 0
            start_foud = False
            stop_foud = False
            i = 0
            while i < len(line.strip()):
                if line[i] in ["-", "i"]:
                    i += 1
                    continue
                if line[i] == "d":
                    which_codon += 1
                    i += 3
                    continue
                if msa_state_seq[i] == "A":
                    start_id = config.model.str_to_state_id("stA")
                    true_state_seq += [start_id, start_id + 1, start_id + 2]
                    start_foud = True
                    i += 3
                elif msa_state_seq[i] == "S":
                    stop_id = config.model.str_to_state_id("stop1")
                    true_state_seq += [stop_id, stop_id + 1, stop_id + 2]
                    stop_foud = True
                    i += 3
                elif msa_state_seq[i] == "i":
                    insert_id = config.model.str_to_state_id(f"i_{which_codon},0")
                    true_state_seq += [insert_id, insert_id + 1, insert_id + 2]
                    i += 3
                else:
                    if stop_foud:
                        true_state_seq += [config.model.str_to_state_id("ig3'")]
                        i += 1
                    elif not start_foud:
                        true_state_seq += [config.model.str_to_state_id("ig5'")]
                        i += 1
                    else:
                        codon_id = config.model.str_to_state_id(f"c_{which_codon},0")
                        true_state_seq += [codon_id, codon_id + 1, codon_id + 2]
                        which_codon += 1
                        i += 3
            if len(true_state_seqs) == 0:
                true_state_seqs = [true_state_seq]
            else:
                true_state_seqs.append(true_state_seq)
    return true_state_seqs

def load_viterbi_guess(config):
    viterbi_file = open(f"{config.current_run_dir}/viterbi_cc_output_{'before' if config.after_or_before == 'b' else 'after'}.json", "r")
    viterbi = json.load(viterbi_file)
    return viterbi

def write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi):
    def state_seq_to_nice_str(state_seq): #[0,0,0,0,0,1,2,3,4,5,6,7,8,9,16,16,16,16]
        s = ""
        for i, state in enumerate(state_seq):
            if state == config.model.str_to_state_id("ig5'"):
                s += "5"
            if state == config.model.str_to_state_id("stA"):
                s += "A"
            if state == config.model.str_to_state_id("stT"):
                s += "T"
            if state == config.model.str_to_state_id("stG"):
                s += "G"
            if state == config.model.str_to_state_id("stop1"):
                s += "S"
            if state == config.model.str_to_state_id("stop2"):
                s += "T"
            if state == config.model.str_to_state_id("stop3"):
                s += "P"
            if state == config.model.str_to_state_id("ig3'"):
                s += "3"
            # insert or codon
            if config.model.state_id_to_str(state)[-1] == "0": # and len(config.model.state_id_to_str(state)) == 5
                splitted = re.split("[_|,]",config.model.state_id_to_str(state))
                s += splitted[0] + (splitted[1] + (" " if len(splitted[1]) < 2 else ""))[:2]
        return s

    # state_str      555ATGc0 STP33
    # msa_seq_str ---ATCATGCTATAGTA-----
    def expand_nice_states_str_to_fit_msa(state_str, msa_seq_str):
        id_in_state_str = 0
        new_str = ""
        for cc in msa_seq_str:
            if cc == "-":
                new_str += "-"
            elif cc == "\n":
                new_str += "\n"
            elif cc in ["i", "d"]:
                new_str += " "
            else:
                new_str += state_str[id_in_state_str]
                id_in_state_str += 1
        return new_str

    with open(f"{config.current_run_dir}/trueMSA.txt","r") as msa:
        with open(f"{config.current_run_dir}/trueMSA_viterbi.txt","w") as out:
            for i, msa_seq_str in enumerate(msa):
                seq_id = i - 1
                if seq_id < 0:
                    out.write("# first line is msa state seq. the first of the 3 tuple is aligned seq, second is true state seq, third is viterbi guess\n")
                    out.write(msa_seq_str)
                    continue
                else:
                    out.write(msa_seq_str) # ----TTATGTTCTAATCGGTT from useMSAgen trueMSA
                    add_one_terminal_symbol = True
                    one = 1 if add_one_terminal_symbol else 0
                    try:
                        x = viterbi[seq_id]
                    except:
                        print("for this seq_id there is no viterbi guess")
                        break
                    assert len(viterbi[seq_id]) - one == len(true_state_seqs[seq_id]), f"length if viterbi {viterbi[seq_id]} and true state seq {true_state_seqs[seq_id]} differ, check if parallel computing works as intended"
                    out.write(expand_nice_states_str_to_fit_msa(state_seq_to_nice_str(true_state_seqs[seq_id]), msa_seq_str))
                    out.write(expand_nice_states_str_to_fit_msa(state_seq_to_nice_str(viterbi[seq_id]), msa_seq_str))
                out.write("\n")

def eval_start_stop(config, viterbi):
    start = time.perf_counter()
    print("starting eval viterbi")

    stats = {"start_not_found" : 0,\
             "start_too_early" : 0,\
             "start_correct" : 0,\
             "start_too_late" : 0,\
             "stop_not_found" : 0,\
             "stop_too_early" : 0,\
             "stop_correct" : 0,\
             "stop_too_late" : 0}

    start_stop = pd.read_csv(f"{config.current_run_dir}/start_stop_pos.txt", sep=";", header=None)
    stA_id = config.model.str_to_state_id("stA")
    stop1_id = config.model.str_to_state_id("stop1")
    for i, state_seq in enumerate(viterbi):
        try:
            viterbi_start = state_seq.index(stA_id)
        except:
            viterbi_start = -1
        try:
            viterbi_stop = state_seq.index(stop1_id)
        except:
            viterbi_stop = -1


        true_start = start_stop.iloc[i,1]
        true_stop = start_stop.iloc[i,2]

        # print(f"true_start = {true_start} vs viterbi_start = {viterbi_start}")
        # print(f"true_stop = {true_stop} vs viterbi_stop = {viterbi_stop}")

        nSeqs = len(viterbi)

        if viterbi_start == -1:
            stats["start_not_found"] += 1/nSeqs
            if viterbi_stop != -1:
                print("found stop but not start")
                quit(1)
        elif viterbi_start < true_start:
            stats["start_too_early"] += 1/nSeqs
        elif viterbi_start == true_start:
            stats["start_correct"] += 1/nSeqs
        else:
            stats["start_too_late"] += 1/nSeqs

        if viterbi_stop == -1:
            stats["stop_not_found"] += 1/nSeqs
        elif viterbi_stop < true_stop:
            stats["stop_too_early"] += 1/nSeqs
        elif viterbi_stop == true_stop:
            stats["stop_correct"] += 1/nSeqs
        else:
            stats["stop_too_late"] += 1/nSeqs


    with open(f"{config.current_run_dir}/statistics.json", "w") as file:
        json.dump(stats, file)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("done evaluating viterbi. it took ", time.perf_counter() - start)

def compare_guess_to_true_state_seq(trues, guesses):
    correct = 0
    false = 0
    for true, guess in zip(trues, guesses):
        for i, (x,y) in enumerate(zip(true, guess)):
            if x != y:
                false += 1
                print("true seq")
                print(true[:i], "__first_mistake__", true[i:])
                print("prediction")
                print(guess[:i], "__first_mistake__", guess[i:])
                break
        else:
            correct += 1
    print(f"correct = {correct}, false = {false}, accuracy = {correct/(false+correct)}")



################################################################################
################################################################################
################################################################################
def retrieve_homo_sapiens_from_combinded_fasta(fasta_path):
        try:
            fasta_data = SeqIO.parse(fasta_path, "fasta")
            for record in fasta_data:
                if re.search("Homo_sapiens", record.id):
                    human_fasta = record
                    return human_fasta
            try:
                human_fasta.id
            except:
                print("fasta_true_state_seq_and_optional_viterbi_guess_alignment: no human id found: return")
                return
        except:
            print("seqIO could not parse", fasta_path)
            return
################################################################################
def read_viterbi_json(model, viterbi_path) -> list[tuple[int,str]]:
        l = []
        if viterbi_path != None:
            assert model != None, "if you want to include the viterbi guess, you also have to provide a model to convert the state seq to a string"

            try:
                file = open(viterbi_path)
            except:
                print("could not open", viterbi_path)
                return
            try:
                json_data = json.load(file)
            except:
                print("json could not parse", file)
                return

            assert len(json_data) == 1, f"more than one seq in {viterbi_path}"

            if type(json_data[0]) is list: #[[0,1,2],[0,0,1],[1,2,3,4,5]]
                description_seq = []
                for seq_id, seq in enumerate(json_data):
                    for nth_state, state in enumerate(seq):
                        description = model.state_id_to_str(state)
                        description_seq.append((state,description))
                    l.append(description_seq)
            # else: # [0,0,0,01,2,3,4,4,4,4]
            #     for nth_state, state in enumerate(json_data):
            #         description = model.state_id_to_str(state)
            #         l.append((state,description))
        return l
################################################################################
def viterbi_tuple_list_to_fasta(viterbi_path, human_fasta, l : list[int,str]) -> str:
    '''return st. like  llll---AGc 0c 1c 2c 3c 4c 5c 6c 7c 8c 9c10c11c12GT---rrrr'''
    viterbi_as_fasta = ""
    if viterbi_path == None:
        viterbi_as_fasta = " " * len(human_fasta.seq)
    else:
        i = 0
        while i < len(l[0]):
            state_id, description = l[0][i]
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
            elif description.startswith("i_"):
                insert_id = re.search(r"i_(\d+),", description).group(1)
                if len(insert_id) > 2:
                    viterbi_as_fasta += "I" # this indicates that insert id is larger than 100, bc only the last two digits can be displayed
                else:
                    viterbi_as_fasta += "i"
                if l[0][i+1][1] == "G":
                    i-=2 # -2 bc adding 2 and 1 anyways
                elif l[0][i+2][1] == "G":
                    viterbi_as_fasta += "."
                    i-=1
                else:
                    viterbi_as_fasta += ("." + insert_id)[-2:]
                i+=2
            elif description.startswith("c_"):
                insert_id = re.search(r"c_(\d+),", description).group(1)
                if len(insert_id) > 2:
                    viterbi_as_fasta += "C" # this indicates that insert id is larger than 100, bc only the last two digits can be displayed
                else:
                    viterbi_as_fasta += "c"
                if l[0][i+1][1] == "G":
                    i-=2
                elif l[0][i+2][1] == "G":
                    viterbi_as_fasta += "."
                    i-=1
                else:
                    viterbi_as_fasta += ("." + insert_id)[-2:]
                i+=2
            else:
                viterbi_as_fasta += "-"

            i += 1

        # removing terminal
        viterbi_as_fasta = viterbi_as_fasta[:-1]
        print("viterbi_as_fasta", viterbi_as_fasta)

        assert l[0][-1][1] == "ter", "Model.py last not terminal"
    return viterbi_as_fasta
################################################################################
def get_true_gene_strucutre_from_fasta_description_as_fasta(coords) -> tuple[str, bool, dict]:
    on_reverse_strand = coords["exon_start_in_human_genome_cd_strand"] != coords["exon_start_in_human_genome_+_strand"]
    if not on_reverse_strand:
        left_intron_len = (coords["exon_start_in_human_genome_+_strand"] - coords["seq_start_in_genome_+_strand"])
        exon_len = (coords["exon_stop_in_human_genome_+_strand"] - coords["exon_start_in_human_genome_+_strand"])
        right_intron_len = (coords["seq_stop_in_genome_+_strand"] - coords["exon_stop_in_human_genome_+_strand"])
    else:
        left_intron_len = (coords["seq_start_in_genome_cd_strand"] - coords["exon_start_in_human_genome_cd_strand"])
        exon_len = (coords["exon_start_in_human_genome_cd_strand"] - coords["exon_stop_in_human_genome_cd_strand"])
        right_intron_len = (coords["exon_stop_in_human_genome_cd_strand"] - coords["seq_stop_in_genome_cd_strand"])
    true_seq = "l" * left_intron_len
    true_seq += "E" * exon_len
    true_seq += "r" * right_intron_len
    d = {"left_intron_len": left_intron_len, \
         "exon_len" : exon_len, \
         "right_intron_len" : right_intron_len}
    return true_seq, on_reverse_strand, d
################################################################################
def get_coords_record(len_of_line_in_clw, coords, seq_len, on_reverse_strand):
    coords_fasta = ""
    for line_id in range(seq_len//len_of_line_in_clw):
        in_fasta = line_id*len_of_line_in_clw
        if not on_reverse_strand:
            coords_line = f"in this fasta {in_fasta}, in genome {in_fasta + coords['seq_start_in_genome_+_strand']}"
        else:
            coords_line = f"in this fasta {in_fasta}, in genome {coords['seq_start_in_genome_cd_strand']- in_fasta}"
        coords_fasta += coords_line + " " * (len_of_line_in_clw - len(coords_line))

    last_line_len = seq_len - len(coords_fasta)
    coords_fasta += " " * last_line_len

    coords_fasta_record = SeqRecord(seq = Seq(coords_fasta), id = "coords_fasta")
    return coords_fasta_record
################################################################################
def get_numerate_line_record(seq_len, len_of_line_in_clw = 50):
    numerate_line = ""
    for i in range(seq_len):
        i_line = i % len_of_line_in_clw
        if i_line % 10 == 0:
            numerate_line += "|"
        else:
            numerate_line += " "

    numerate_line_record = SeqRecord(seq = Seq(numerate_line), id = "numerate_line")
    return numerate_line_record
################################################################################
def fasta_true_state_seq_and_optional_viterbi_guess_alignment(fasta_path, viterbi_out_path = None, model = None, out_dir_path = "."):
    '''
    fasta_path is config.fasta_path (combined.fasta in most cases)
    assumes viterbi only contains prediction for human
    '''
    len_of_line_in_clw = 50
    # TODO: maybe also implement model.state_id_to_description_single_letter()

    if (human_fasta := retrieve_homo_sapiens_from_combinded_fasta(fasta_path)) is None:
        return

    coords = json.loads(re.search("({.*})", human_fasta.description).group(1))

    if viterbi_out_path is not None:
        l = read_viterbi_json(model, viterbi_out_path)
        viterbi_as_fasta = viterbi_tuple_list_to_fasta(viterbi_out_path, human_fasta, l)

        # out_viterbi_file_path = f"{config.current_run_dir}/viterbi_cc_output_{after_or_before}.json"
        viterbi_as_fasta_path = re.sub("_cc_output", "", viterbi_out_path)
        viterbi_as_fasta_path = re.sub(".json", ".fasta", viterbi_as_fasta_path)

        viterbi_record = SeqRecord(seq = Seq(viterbi_as_fasta), id = "viterbi_guess")
        with open(viterbi_as_fasta_path, "w") as viterbi_fa_file:
            SeqIO.write(viterbi_record, viterbi_fa_file, "fasta")


    true_seq, on_reverse_strand, _ = get_true_gene_strucutre_from_fasta_description_as_fasta(coords)
    true_seq_record = SeqRecord(seq = Seq(true_seq), id = "true_seq")

    numerate_line_record = get_numerate_line_record(len(true_seq), len_of_line_in_clw = len_of_line_in_clw)

    coords_fasta_record = get_coords_record(len_of_line_in_clw, coords, len(true_seq), on_reverse_strand)

    exon_contains_ambiguous_bases = ""
    for base, e_or_i in zip(human_fasta.seq, true_seq_record.seq):
        if e_or_i == "E" and base in "acgtnN":
            exon_contains_ambiguous_bases = "_exon_contains_ambiguous_bases"

    if viterbi_out_path == None:
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record]
    else:
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record, viterbi_record]

    alignment = MultipleSeqAlignment(records)

    after_or_before = ""
    if viterbi_out_path != None:
        print(viterbi_out_path)

        found_after = re.search("after", os.path.basename(viterbi_out_path))
        found_before = re.search("before", os.path.basename(viterbi_out_path))

        assert not (found_after and found_before), "found both 'after' and 'before' in viterbi_cc_ file name"
        assert found_after or found_before, "found no 'after' or 'before' in viterbi_cc_ file name"

        after_or_before = "_after" if found_after else "_before"
        print(after_or_before)

    alignment_out_path = f"{out_dir_path}/true_alignment{after_or_before}{exon_contains_ambiguous_bases}.clw"

    with open(alignment_out_path, "w") as output_handle:
        AlignIO.write(alignment, output_handle, "clustal")
    print("wrote alignment to", alignment_out_path)

################################################################################
################################################################################
################################################################################
def run_cc_viterbi(config, matr_dir):
    import multiprocessing
    start = time.perf_counter()
    seq_path = f"--seqs_path {config.fasta_path}.json"
    only_first_seq = f"--only_first_seq" if config.only_first_seq else ""
    # if config.manual_passed_fasta:
    #     out_dir_path = os.path.dirname(config.fasta_path)
    #     out_path = f"--out_path {out_dir_path}/viterbi_cc_output.json"
    # else:
    #     out_path = ""

    after_or_before = os.path.basename(matr_dir).split("_")[0]
    assert after_or_before in ["before", "after"], "Viterbi.py. Folder of matrices must begin with either 'before' or 'after'"
    out_path = f"--out_path {config.current_run_dir}/viterbi_cc_output_{after_or_before}.json"
    # command = f"{config.out_path}/Viterbi -c {config.nCodons} -j {multiprocessing.cpu_count()-1} {seq_path} {only_first_seq} {out_path}"
    print("matr_dir passed to cc viterbi", matr_dir)

    # finding the Viterbi exe
    command = f"{config.viterbi_exe} -j {config.viterbi_threads} {seq_path} {only_first_seq} {out_path} --dir_path_for_para {matr_dir}"
    print("starting", command)
    os.system(command)
    print("done viterbi. it took ", time.perf_counter() - start)
################################################################################
def main(config, overwrite_viterbi = True):
    # (if not passed parent_input_dir it is set to current run dir but this is different when calling viterbi
    # when parent_input_dir is not set and viterby.py is called d viterbi.py should select the newest dir)

    # check if matrices exists, if not convert kernels

    after_or_before = "after_fit_para" if config.after_or_before == "a" else "before_fit_para"
    matr_dir = f"{config.current_run_dir}/{after_or_before}"
    # if not os.path.exists(f"{matr_dir}/A.json"):

    # TODO maybe dont convert if files already exist
    print("start converting kernels to matrices")
    convert_kernel_files_to_matrices_files(config, matr_dir)
    print("done with converting kernels")

    seqs_json_path = f"{config.fasta_path}.json"
    if not os.path.exists(seqs_json_path):
        print("fa.json doesnt exist, so it it calculated")
        from ReadData import read_data_one_hot_with_Ns_spread_str
        from ReadData import convert_data_one_hot_with_Ns_spread_str_to_numbers
        seqs = read_data_one_hot_with_Ns_spread_str(config, add_one_terminal_symbol = True)
        seqs_out = convert_data_one_hot_with_Ns_spread_str_to_numbers(seqs)
        with open(seqs_json_path, "w") as out_file:
            json.dump(seqs_out, out_file)
        print("finished calculating fa.json")

    after_or_before = os.path.basename(matr_dir).split("_")[0]
    assert after_or_before in ["before", "after"], "Viterbi.py. Folder of matrices must begin with either 'before_' or 'after_'"

    # if viteribi is already calculated, path to viterbi file which is then written to the alignment
    if config.in_viterbi_path:
        assert config.manual_passed_fasta, "when viterbi.py and --in_viterbi_path also pass --fasta"
        fasta_true_state_seq_and_optional_viterbi_guess_alignment(config.fasta_path, config.in_viterbi_path, config.model, out_dir_path = config.current_run_dir)

    out_viterbi_file_path = f"{config.current_run_dir}/viterbi_cc_output_{after_or_before}.json"
    if not config.in_viterbi_path:
        if os.path.exists(out_viterbi_file_path):
            if overwrite_viterbi:
                run_cc_viterbi(config, matr_dir)
            else:
                print(f"{out_viterbi_file_path} already exists and --force_overwrite was not passed")
        else:
            run_cc_viterbi(config, matr_dir)

        # if config.manual_passed_fasta:
        print("before fasta_true_state_seq_and_optional_viterbi_guess_alignment")
        fasta_true_state_seq_and_optional_viterbi_guess_alignment(config.fasta_path, out_viterbi_file_path, config.model, out_dir_path = config.current_run_dir)
        print("adter fasta_true_state_seq_and_optional_viterbi_guess_alignment")
        # if not config.manual_passed_fasta:
        #     viterbi_guess = load_viterbi_guess(config)
        #     true_state_seqs = get_true_state_seqs_from_true_MSA(config)
        #     compare_guess_to_true_state_seq(true_state_seqs, viterbi_guess)
        #     write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi_guess)
        #     eval_start_stop(config, viterbi_guess)


if __name__ == "__main__":

    from Config import Config

    print("started making config in viterbi.py")
    config = Config()
    config.init_for_viterbi()
    print("done with making config in vertbi.py")

    # print("viterbi with after_fit_matrices or before_fit_matrices [a/b]")
    # while (a_or_b := input()) not in "ab":
    #     pass

    main(config)

