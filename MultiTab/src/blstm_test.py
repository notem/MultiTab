
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


    get_sample = lambda: generate_sample(args, args.up, args.lp, samples, classes, counts=1, **process_kwargs)

    correct = 0
    wrong = 0

    time_distances = []
    random_time_distances = []
    trandom_time_distances = []
    s = time()
    for i in range(2000):
        X,y,sp,X_raw = get_sample()[0]
        x_tensor = np.array(X)[np.newaxis, ...]
        x_batch = torch.tensor(x_tensor, dtype=torch.float32)#.squeeze(-1)
        true_time = X_raw[0][sp]

        model.init_hidden(x_batch.size(0))
        output = model(x_batch).detach().numpy()

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
        if abs(guess - sp) <= 25:
            correct += 1
        else:
            wrong += 1

        #guess_time = (guess*0.5)
        guess_time = X_raw[0][guess]
        time_distance = abs(guess_time - true_time)
        time_distances.append(time_distance)

        # random guess w/ assumed knowledge
        random_time = random.uniform(2, min(X_raw[0][-1],30))
        random_time_dist = abs(true_time - random_time)
        random_time_distances.append(random_time_dist)

        # random guess w/ no knowledge
        trandom_time = random.uniform(0, X_raw[0][-1])
        trandom_time_dist = abs(true_time - trandom_time)
        trandom_time_distances.append(trandom_time_dist)

        print(f"[{i}] {correct/(wrong+correct)} Model= dist: {time_distance}, avg_dist: {np.mean(time_distances)} Random= dist: {random_time_dist}, avg_dist: {np.mean(random_time_distances)}", end='\r')
    print("")

    e = time()
    print(f"Finished testing in {e-s} seconds.")
    print(f"Modeled acheived {correct/(wrong+correct)} accuracy")

    fig1, ax1 = plt.subplots()
    data = [time_distances, random_time_distances]
    ax1.boxplot(data, labels=["BLSTM", "Random"])
    plt.savefig('boxplot_blstm.png')
    #np.save('blstm_time_dist.npy')
    #np.save('random1_time_dist.npy')

if __name__ == "__main__":
    main()
