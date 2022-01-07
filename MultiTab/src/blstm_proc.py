
import torch

import pickle
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from dataprep import load_data, generate_multitab_time
import random

from blstm import generate_sample, process, do_slices, BLSTM, CNNBLSTM
import matplotlib.pyplot as plt
from time import time


def main():

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)  

    parser = ArgumentParser()
    parser.add_argument("-ld", required=True)
    parser.add_argument("-sd", required=True)
    parser.add_argument("-m", required=True)
    args = parser.parse_args()

    n_features = 1
    #process_kwargs = {"slice_size": n_features, "style": "time-bursts", "interval": 0.5, "slice_size":n_features, "stride_size":n_features}
    process_kwargs = {"style": "tiktok", "slice_size":n_features, "stride_size":n_features}

    model = CNNBLSTM(n_features).to(device)
    model.load_state_dict(torch.load(args.m))
    model.eval()
    print(model)

    print("+ loading data")
    with open(args.ld, 'rb') as fi:
        samples = pickle.load(fi)
    classes = list(np.unique(samples['y']))

    cleaned_X = []

    s = time()
    for i in range(len(samples['X'])):
        print(f"{i}/{len(samples['X'])}", end='\r')
        X_raw = samples['X'][i]

        X = process(X_raw, **process_kwargs)
        sp = samples['p'][0][0]
        slices, y = do_slices(X, process_kwargs["slice_size"], process_kwargs["stride_size"], sp)
        #y = np.zeros(len(slices))
        #y[start_pkt_idx:] = 1

        x_tensor = np.array(X)[np.newaxis, ..., np.newaxis]
        x_batch = torch.tensor(x_tensor, dtype=torch.float32).to(device)#.squeeze(-1)

        model.init_hidden(1, device=device)
        output = model(x_batch).cpu().detach().numpy()

        preds = output > 0.5
        guess = len(preds)
        threshold = 50
        for i in range(threshold, len(preds)):
            tmp = True
            for j in range(threshold):
                if not preds[i-threshold+j]:
                    tmp = False
                    break
            if tmp:
                guess = i-threshold
                break
        #pkt_cnt = 0
        #for i in range(x_tensor.shape[0], guess):
        #    pkt_cnt += np.sum(np.abs(X_tensor[i]))
        #if abs(pkt_cnt - sp) <= 25:
        #    correct += 1
        #else:
        #    wrong += 1

        cleaned_X.append(X_raw[:,:guess])

    e = time()
    print(f"Finished processing samples.")

    with open(args.sd, 'wb') as fi:
        pickle.dump({'X': cleaned_X, 'y': samples['y']}, fi)

if __name__ == "__main__":
    main()
