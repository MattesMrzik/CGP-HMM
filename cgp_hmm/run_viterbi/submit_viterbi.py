#!/usr/bin/env python3
import os
import argparse
import re
parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', required=True, help='number of codons')
# parser.add_argument('-n', default = 1, type = int, help='number of Nodes and as arg to viterbi.cc')
parser.add_argument('-mpi', default = 1, type = int, help='number of mpi processes or should this be arg to viterbi.cc')
parser.add_argument('--seqs_path', default =  "/home/s-mamrzi/CGP-HMM/cgp_hmm/output/{args.c}codons/seqs.fa.json", help = 'optional path to seqs.fa')
parser.add_argument('--i_path', default = "/home/s-mamrzi/CGP-HMM/cgp_hmm/output/{args.c}codons/after_fit_matrices/I.json", help = 'optional path to I.json')
parser.add_argument('--a_path', default = "/home/s-mamrzi/CGP-HMM/cgp_hmm/output/{args.c}codons/after_fit_matrices/A.json", help = 'optional path to A.json')
parser.add_argument('--b_path', default = "/home/s-mamrzi/CGP-HMM/cgp_hmm/output/{args.c}codons/after_fit_matrices/B.json", help = 'optional path to B.json')
parser.add_argument('--out_path', default = "/home/s-mamrzi/CGP-HMM/cgp_hmm/output/{args.c}codons/viterbi_cc_output.json", help = 'optional out path')
parser.add_argument('--only_first_seq', action = 'store_true', help = 'only_first_seq')



args = parser.parse_args()
args.seqs_path = re.sub("{args.c}",args.c,args.seqs_path)
args.i_path = re.sub("{args.c}",args.c,args.i_path)
args.a_path = re.sub("{args.c}",args.c,args.a_path)
args.b_path = re.sub("{args.c}",args.c,args.b_path)
args.out_path = re.sub("{args.c}",args.c,args.out_path)
args.n = 1

args.out_path = "viterbi_cc_output.json"

submission_file_name = "submission_file.sh"
os.system(f"rm {submission_file_name}")
os.system("rm err* out*")

with open(submission_file_name, "w") as file:
    file.write("#!/bin/bash\n")
    file.write("#SBATCH -J viterbi\n")
    file.write(f"#SBATCH -N {args.n}\n")
    file.write(f"#SBATCH -n {args.mpi}\n")
    file.write("#SBATCH --partition=compute\n")
    file.write("#SBATCH -o out.%j\n")
    file.write("#SBATCH -e err.%j\n")
    file.write("#SBATCH -t 02:00:00\n")
    file.write("#SBATCH --mail-type=end\n")
    file.write("#SBATCH --mail-type=end\n")
    # oder ist anzahl der processe liever mpi, oder mpi mal n?
    file.write(f"/home/s-mamrzi/CGP-HMM/cgp_hmm/Viterbi -c {args.c} -j {args.mpi} --seqs_path {args.seqs_path} --i_path {args.i_path} --a_path {args.a_path} --b_path {args.b_path} --out_path {args.out_path} {'--only_first_seq' if args.only_first_seq else ''} \n")
    # file.write("touch job_done")

os.system("echo -------------------------------")
os.system(f"cat {submission_file_name}")
os.system("echo -------------------------------")

os.system(f"sbatch {submission_file_name}")
#while not os.path.exists("job_done"):
#    os.system("squeue | grep s-mamrzi")
#    # maybe also print .out and .err
#os.system("rm job_done")
