export CUDA_VISIBLE_DEVICES=0
python blstm_proc.py -ld data_1_10_upto4.npy -sd lstm_data_2_10_upto4.npy -m model5.pt
python blstm_proc.py -ld data_2_10_2tabs.npy -sd lstm_data_2_10_2tabs.npy -m model5.pt
python blstm_proc.py -ld data_2_10_3tabs.npy -sd lstm_data_2_10_3tabs.npy -m model5.pt
python blstm_proc.py -ld data_2_10_4tabs.npy -sd lstm_data_2_10_4tabs.npy -m model5.pt
