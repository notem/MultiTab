import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool
from itertools import repeat


def intercell_times(cells):
    """
    """
    times = []
    for i in range(1, len(cells)):
        times.append(abs(cells[i][0] - cells[i-1][0]))
    return times

def cell_features(candidate, before, after):
    """
    """
    features = []

    before_ic_times = intercell_times(before)
    if len(before) > 0:
        before_ic_times += [abs(candidate[0] - before[-1][0])]
    after_ic_times = intercell_times(after)
    if len(after) > 0:
        after_ic_times = [abs(candidate[0] - after[0][0])] + after_ic_times

    # PROXIMITY INTERCELL TIMES (5)
    f = [before_ic_times[-i] if i < len(before_ic_times) else None for i in range(0, 3, 1)]
    features.extend(f)
    f = [after_ic_times[i] if i < len(after_ic_times) else None for i in range(0, 2, 1)]
    features.extend(f)
    
    # FIFTY CELL STATS (3)
    f = [np.mean(after_ic_times+before_ic_times), np.std(after_ic_times+before_ic_times), np.amax(after_ic_times+before_ic_times)]
    features.extend(f)

    # CANDIDATE TIME DIFFERENCE (11)
    f = [abs(candidate[0] - before[0][0]) if len(before) > 0 else None]
    features.extend(f)
    f = [abs(after[0][0] - candidate[0]) if len(after) > 0 else None]
    features.extend(f)
    f = [abs(after[i][0] - before[-i][0]) if (i < len(before) and i < len(after)) else None for i in range(2, 19, 2)]
    features.extend(f)

    # CELL COUNT (4)
    tmp = []
    f = [sum([1 if (i < len(before) and before[-i][1] > 0) else 0 for i in range(5)])] if len(before) > 0 else [0]
    tmp.extend(f)
    f = [sum([1 if (i < len(after) and after[i][1] > 0) else 0 for i in range(5)])] if len(after) > 0 else [0]
    tmp.extend(f)
    f = [tmp[-1], tmp[-2]]
    features.extend(tmp)
    f = [sum([1 if (i < len(before) and before[-i][1] > 0) else 0 for i in range(10)])] if len(before) > 0 else [0]
    tmp.extend(f)
    f = [sum([1 if (i < len(after) and after[i][1] > 0) else 0 for i in range(10)])] if len(after) > 0 else [0]
    tmp.extend(f)
    f = [tmp[-1], tmp[-2]]
    features.extend(tmp)

    return features

def generate_features(trace, proximity_size=50, replace_none=-1, samples=200):
    """
    """
    point_features = []
    idx = list(range(0, len(trace[0])))
    if samples > 0 and len(idx) > samples:
        idx = np.random.choice(idx, samples, replace=False)
    for i in idx:
        candidate = (trace[0][i], trace[1][i])
        before = [(trace[0][i-j], trace[1][i-j]) for j in range(1,1+proximity_size) if i-j >= 0]
        after = [(trace[0][i+j], trace[1][i+j]) for j in range(1,1+proximity_size) if i+j < len(trace[0])]
        features = cell_features(candidate, before, after)
        for i in range(len(features)):
            if features[i] is None:
                features[i] = replace_none
        point_features.append(features)
    return point_features

def main():
    parser = ArgumentParser()
    parser.add_argument("-sd", required=True)
    parser.add_argument("-ld", required=True)
    args = parser.parse_args()

    print(f"+ loading samples from {args.ld}")
    with open(args.ld, 'rb') as fi:
        samples = pickle.load(fi)
    X_raw = samples['X'][:50000]
    y_raw = samples['y'][:50000]
    p = samples['p'][:50000]
    print(len(p), len(X_raw))

    print("+ processing into features")
    proximity_size = 50
    X, y = [], []
    for i in tqdm(range(5000)):
        # generate true split features
        split_idx = p[i][0]
        if split_idx < 0 or split_idx >= len(X_raw): continue
        candidate = (X_raw[i][0][split_idx], X_raw[i][1][split_idx])
        before = [(X_raw[i][0][split_idx-j], X_raw[i][1][split_idx-j]) for j in range(1,1+proximity_size) if split_idx-j >= 0]
        after = [(X_raw[i][0][split_idx+j], X_raw[i][1][split_idx+j]) for j in range(1,1+proximity_size) if split_idx+j < len(X_raw[i][0])]
        f = cell_features(candidate, before, after)
        for i in range(len(f)):
            if f[i] is None:
                f[i] = -1
        X.append(f)
        y.append(1)
    with Pool() as pool:
        #it = pool.starmap(generate_features, (X_raw[:30000], repeat(proximity_size), repeat(-1), repeat(30)), chunksize=10)
        it = pool.imap(generate_features, X_raw[:5000], chunksize=10)
        for i, features in tqdm(enumerate(it), total=5000):
            labels = [0]*len(features)
            X.extend(features)
            y.extend(labels)
    
    X = np.array(X)
    y = np.array(y)

    print(f"+ saving features and labels to {args.sd}")
    with open(args.sd, 'wb') as fi:
        pickle.dump({'X_raw': X_raw[5000:7000], 'p': p[5000:7000], 'X': X, 'y': y}, fi)

if __name__ == "__main__":
    main()
