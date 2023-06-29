
## Install

* git clone --recursive https://github.com/MattesMrzik/CGP-HMM
* cd viterbi_cc
* make
* chmod u+x Viterbi

## Run

* cd src
* python3 cgphmm.py

## Dataset creation
* _get\_exons\_df.py_ creates a df containing all exons.
* _get\_internal\_exon.py_ selects suitable exons. Mapps coordinates and extracts fasta sequences.
* _select\_good\_exons\_for\_training.py_ selects exons that can be used for training. Copies them to a new directory.

## Evaluation
* the script _multi\_run.py_ was used for running and evaluation.