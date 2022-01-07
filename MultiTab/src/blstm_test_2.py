
import torch

import pickle
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from dataprep import load_data, generate_multitab_time
import random

from blstm import generate_sample, process, BLSTM, CNNBLSTM
import matplotlib.pyplot as plt
from time import time


def main():

    parser = ArgumentParser()
    parser.add_argument("-ld", required=True)
    parser.add_argument("-tc", required=False, default=2, type=int)
    parser.add_argument("-lp", required=False, default=2, type=float)
    parser.add_argument("-up", required=False, default=30, type=float)
    parser.add_argument("-m", required=True)
    args = parser.parse_args()

    n_features = 1
    #process_kwargs = {"slice_size": n_features, "style": "time-bursts", "interval": 0.5, "slice_size":n_features, "stride_size":n_features}
    process_kwargs = {"style": "tiktok", "slice_size":n_features, "stride_size":n_features}

    model = CNNBLSTM(n_features)
    model.load_state_dict(torch.load(args.m))
    model.eval()

    print("+ loading data")
    samples = load_data(args.ld)
    classes = list(samples.keys())

    #process_kwargs = {"slice_size": n_features, "style": "time-bursts", "interval": 0.5, "slice_size":n_features, "stride_size":n_features}


    get_sample = lambda t: generate_sample(args, t+1, t, samples, classes, counts=1, **process_kwargs)


    pkt_ths = np.arange(1,2000)
    corr_th = [0]*len(pkt_ths)
    wron_th = [0]*len(pkt_ths)
    accs = []
    dists = []
    s = time()
    for t in range(args.lp, args.up):
        time_distances = []
        correct = 0
        wrong = 0
        for i in range(20):
            X,y,sp,X_raw = get_sample(t)[0]
            x_tensor = np.array(X)[np.newaxis, ...]
            x_batch = torch.tensor(x_tensor, dtype=torch.float32)#.squeeze(-1)
            true_time = X_raw[0][sp]

            model.init_hidden(x_batch.size(0))
            output = model(x_batch).detach().numpy()

            preds = output > 0.5
            guess = len(preds)
            threshold = 50
            for z in range(threshold, len(preds)):
                tmp = True
                for j in range(threshold):
                    if not preds[z-threshold+j]:
                        tmp = False
                        break
                if tmp:
                    guess = z-threshold
                    break
            #pkt_cnt = 0
            #for i in range(x_tensor.shape[0], guess):
            #    pkt_cnt += np.sum(np.abs(X_tensor[i]))
            #if abs(pkt_cnt - sp) <= 25:
            #    correct += 1
            #else:
            #    wrong += 1
            for idx,pkt_th in enumerate(pkt_ths):
                if abs(guess - sp) <= pkt_th:
                    corr_th[idx] += 1
                else:
                    wron_th[idx] += 1

            if abs(guess - sp) <= 25:
                correct += 1
            else:
                wrong += 1

            #guess_time = (guess*0.5)
            guess_time = X_raw[0][guess]
            time_distance = abs(guess_time - true_time)
            time_distances.append(time_distance)

            print(f"[{i}|{t}] {correct/(wrong+correct)} Model= dist: {time_distance}, avg_dist: {np.mean(time_distances)}", end='\r')
        dists.append(time_distances)
        accs.append(correct/(wrong+correct))
        #np.save(f'timedist.npy', dists)
        #np.save(f'accs.npy', accs)
        np.save(f'acc_v_th.npy', [correct/(correct+wrong) for correct,wrong in zip(corr_th,wron_th)])
        print("")

    np.save(f'timedist.npy', dists)
    np.save(f'accs.npy', accs)
    np.save(f'acc_v_th.npy', [correct/(correct+wrong) for correct,wrong in zip(corr_th,wron_th)])

    e = time()
    print(f"Finished testing in {e-s} seconds.")


    #fig1, ax1 = plt.subplots()
    #data = [time_distances, random_time_distances]
    #ax1.boxplot(data, labels=["BLSTM", "Random"])
    #plt.savefig('boxplot_blstm.png')

if __name__ == "__main__":
    main()
