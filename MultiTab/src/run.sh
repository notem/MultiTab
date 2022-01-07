python dataprep.py -sd multitab_time.pkl -ld /home/shared-data/datasets/AWF100/awf_cw/ -up 10 -lp 5
python wang_features.py -sd features_time.pkl -ld multitab_time.pkl
python split_finding.py -ld features_time.pkl
