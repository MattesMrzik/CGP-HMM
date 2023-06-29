
## Install

* git clone --recursive https://github.com/MattesMrzik/CGP-HMM
* cd viterbi_cc
* make
* chmod u+x Viterbi

## Run

* cd src
* python3 cgphmm.py

## Info
Unit tests are not up to date.

## Dataset creation
* _get_exons_df.py_ creates a df containing all exons.
* _get_internal_exon.py_ selects suitable exon. Mapps coordinates and extraxts fasta sequences.
* _select_good_exons_for_training.py_ selects exons that can be used for training. Copies them to a new directory.