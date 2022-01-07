from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from imblearn.ensemble import RUSBoostClassifier

import torch
import torch.optim as optim
from torch.autograd import Variable

import pickle
import numpy as np
from argparse import ArgumentParser
from wang_features import generate_features
from tqdm import tqdm
import matplotlib.pyplot as plt


def oversample_minority(X, y):
    idxx = [np.where(y==i)[0] for i in np.unique(y)]
    m = max([len(idx) for idx in idxx])
    X_oversample = []
    y_oversample = []
    for idx in idxx:
        c = 0 
        while len(idx)+c < m:
            i = np.random.choice(idx)
            X_oversample.append(X[i])
            y_oversample.append(y[i])
            c += 1
    X = np.concatenate((X, np.array(X_oversample)))
    y = np.concatenate((y, np.array(y_oversample)))

    return X, y


def main():

    parser = ArgumentParser()
    parser.add_argument("-ld", required=True)
    args = parser.parse_args()

    print(f"+ loading features from {args.ld}")
    with open(args.ld, 'rb') as fi:
        samples = pickle.load(fi)
    X_raw = samples['X_raw']
    p = samples['p']
    X = samples['X']
    y = samples['y']

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    X_tr = X[:int(len(y)*0.7)]
    y_tr = y[:int(len(y)*0.7)]
    X_te = X[int(len(y)*0.7):]
    y_te = y[int(len(y)*0.7):]

    #weights = learn_weights(X_tr, y_tr).T
    #print(weights)
    #X_tr *= weights
    #X_te *= weights

    #X_tr,y_tr = oversample_minority(X_tr,y_tr)
    X_te,y_te = oversample_minority(X_te,y_te)

    #idx = np.arange(len(y_tr))
    #np.random.shuffle(idx)
    #X_tr = X_tr[idx][:300000]
    #y_tr = y_tr[idx][:300000]
    #idx = np.arange(len(y_te))
    #np.random.shuffle(idx)
    #X_te = X_te[idx][:20000]
    #y_te = y_te[idx][:20000]

    print(f"+ X_tr {len(X_tr)}, X_te {len(X_te)}, y_tr {sum(y_tr)/len(y_tr)}, y_te {sum(y_te)/len(y_te)}")

    lens = [len(x) for x in X_tr]

    n = 15
    print(f"+ training KNN model")
    print(f"+ n_neighbors={n}")
    #scaler = preprocessing.StandardScaler().fit(X_tr)
    #X_tr = scaler.transform(X_tr)
    #X_te = scaler.transform(X_te)
    #X_tr = preprocessing.normalize(X_tr)
    #X_te = preprocessing.normalize(X_te)

    #clf = KNeighborsClassifier(n_neighbors=n, n_jobs=-1, weights='distance', p=1)
    clf = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R')
    #clf = RandomForestClassifier(n_estimators=n)
    #clf = MLPClassifier(hidden_layer_sizes=(30, 30), early_stopping=True, verbose=1)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0, verbose=1)
    clf = clf.fit(X_tr, y_tr)
    print(f"+ TR_score={clf.score(X_tr, y_tr)}, TE_score={clf.score(X_te, y_te)}")


    time_differences = []
    correct, wrong = 0, 0
    correct_r, wrong_r = 0, 0
    with tqdm(total=len(X_raw)) as pbar:
        for raw, label in zip(X_raw, p):
            candidate_features = generate_features(raw, samples=-1)
            #candidate_features = weights * candidate_features
            #candidate_features = scaler.transform(candidate_features)
            #candidate_features = preprocessing.normalize(candidate_features)
            predictions = clf.predict_proba(candidate_features)[:,1]
            guess = np.argmax(predictions)
            #print(f"pred_range ({np.amin(predictions)}-{np.amax(predictions)}) label {label} true_split {predictions[label]}")
            if abs(label - guess) <= 25:
                correct += 1
            else:
                wrong += 1
            guess_r = np.random.randint(0, len(predictions))
            if abs(label - guess_r) <= 25:
                correct_r += 1
            else:
                wrong_r += 1
            pbar.set_description(f"Acc. {correct/(wrong+correct)}|{correct_r/(wrong_r+correct_r)}")
            pbar.update(1)

            time_pred = raw[0][guess]
            time_true = raw[0][label]
            time_differences.append(abs(time_pred - time_true))

    print(f"Acc (Random guessing): {correct_r/(wrong_r+correct_r)}")
    print(f"Acc (KNN model): {correct/(wrong+correct)}")

    fig1, ax1 = plt.subplots()
    data = [np.array(time_differences).flatten()]
    ax1.boxplot(data, labels=["Wang-RUSBoost"])
    plt.savefig('boxplot_wang.png')

if __name__ == "__main__":
    main()
