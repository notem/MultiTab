import os
from matplotlib import pyplot as plt
import numpy as np
import re
from argparse import ArgumentParser
from tqdm import tqdm
import pickle


def load_trace(fi, separator="\t", filter_by_size=False):
    """
    loads data to be used for predictions
    """
    sequence = [[], [], []]
    for line in fi:
        pieces = line.strip("\n").split(separator)
        if int(pieces[1]) == 0:
            break
        timestamp = float(pieces[0])
        length = abs(int(pieces[1]))
        direction = int(pieces[1]) // length
        if filter_by_size:
            if length > 512:
                sequence[0].append(timestamp)
                sequence[1].append(length)
                sequence[2].append(direction)
        else:
            sequence[0].append(timestamp)
            sequence[1].append(length)
            sequence[2].append(direction)
    return sequence


def generate_multitab_time(traces, offsets):
    get_time_overlap = lambda i: offsets[i] if hasattr(type(offsets), '__iter__') else offsets
    sim_trace = [traces[0][0].copy(), traces[0][1].copy(), traces[0][2].copy()]
    time_offset = 0
    time_merges = []
    split_points = []
    for i in range(len(traces)-1):
        z = len(sim_trace[0])
        time_offset += get_time_overlap(i)
        time_merges.append(time_offset)
        tmerge_point = -1
        for j in range(len(traces[i][0])):
            if traces[i][0][j] > time_offset:
                tmerge_point = j
                break
        sim_trace[0].extend([timestamp+time_offset for timestamp in traces[i+1][0]])
        sim_trace[1].extend(traces[i+1][1])
        sim_trace[2].extend(traces[i+1][2])
        indxs = np.argsort(sim_trace[0])
        sim_trace = np.array([np.array(sim_trace[0])[indxs].tolist(),
                              np.array(sim_trace[1])[indxs].tolist(), 
                              np.array(sim_trace[2])[indxs].tolist()])
        time = time_offset+traces[i+1][0][0]
        merge_point = np.argwhere(sim_trace[0] == time)[0]
        split_points.append(merge_point)
    return sim_trace, np.array(split_points)


def generate_multitab(traces, perc_overlap):
    get_perc_overlap = lambda i: perc_overlap[i] if hasattr(type(perc_overlap), '__iter__') else perc_overlap
    sim_trace = [traces[0][0].copy(), traces[0][1].copy(), traces[0][2].copy()]
    time_offset = 0
    time_merges = []
    split_points = []
    for i in range(len(traces)-1):
        z = len(sim_trace[0])
        perc_overlap = get_perc_overlap(i)
        merge_point = int(z*perc_overlap)
        time_offset += traces[i][0][merge_point]
        time_merges.append(time_offset)
        split_points.append(merge_point)
        sim_trace[0].extend([timestamp+time_offset for timestamp in traces[i+1][0]])
        sim_trace[1].extend(traces[i+1][1])
        sim_trace[2].extend(traces[i+1][2])
        indxs = np.argsort(sim_trace[0])
        sim_trace = np.array([np.array(sim_trace[0])[indxs].tolist(),
                              np.array(sim_trace[1])[indxs].tolist(), 
                              np.array(sim_trace[2])[indxs].tolist()])
    return sim_trace, np.array(split_points)


AWF_PATTERN = r"[\/]([^\/]*)[\/](\d+)$"
DF_PATTERN = r"(\d+)-(\d+)$"
def load_data(root_path, file_pattern=DF_PATTERN, sample_cutoff=100):
    samples = {}
    matcher = re.compile(file_pattern, flags=re.MULTILINE)
    pths = []
    for rt,drs,fnames in os.walk(root_path):
        pths.extend([os.path.join(rt,fname) for fname in fnames])
    for pth in tqdm(pths):
        res = matcher.search(pth)
        if res and res.group(1) and res.group(2):
            cls = res.group(1)
            ins = res.group(2)
            if cls not in samples.keys():
                samples[cls] = []
            if len(samples[cls]) < sample_cutoff:
                with open(pth, 'r') as fi:
                    samples[cls].append(load_trace(fi))
    return samples


def main():
    parser = ArgumentParser()
    parser.add_argument("-sd", required=True)
    parser.add_argument("-ld", required=True)
    parser.add_argument("-tc", required=False, default=2, type=int)
    parser.add_argument("-sc", required=False, default=10000, type=int)
    parser.add_argument("-lp", required=False, default=2, type=float)
    parser.add_argument("-up", required=False, default=30, type=float)
    args = parser.parse_args()

    print("+ loading data")
    samples = load_data(args.ld)
    classes = list(samples.keys())
    sample_leave_out = 0
    
    print("+ generating samples")
    X, y, p = [], [], []
    with tqdm(total=args.sc) as pbar:
        while len(X) < args.sc:
            tab_idxs = np.array(classes)[np.random.randint(0, len(classes), size=args.tc)]
            tabs = []
            for tab_idx in tab_idxs:
                sample = samples[tab_idx][np.random.randint(sample_leave_out, 
                                                            len(samples[tab_idx]))]
                tabs.append(sample)
            if tabs[0][0][-1]+1 < args.lp: continue
            sim_sample, split_points = generate_multitab_time(tabs, np.random.uniform(args.lp, min(args.up, tabs[0][0][-1]), size=len(tabs)-1))
            X.append(sim_sample)
            y.append(1)
            p.append(split_points)
            pbar.update(1)

    with open(args.sd, 'wb') as fi:
        pickle.dump({'X': X, 'y': y, 'p': p}, fi)
    print(f"+ saved to {args.sd}")

if __name__ == "__main__":
    main()
