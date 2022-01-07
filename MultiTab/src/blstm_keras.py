from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from dataprep import load_data, generate_multitab_time
from argparse import ArgumentParser
import numpy as np


def process(sample, slice_size=10, style='directional-bursts', **kwargs):
    X = None
    if style == "directional-bursts":
        X = [0, 0]
        directions = sample[1]
        if directions[0] > 0: 
            X[0] += 1
        else: 
            X[1] += 1
        for i in range(1, len(directions)):
            if directions[i] > 0: 
                if directions[i-1] == directions[i]:
                    X.extend([0,0])
                X[-2] += 1
            else:
                X[-1] += 1

    if style == "iat-bursts":
        X = [0, 0]
        direction = trace[2][0]
        if direction > 0:
            X[0] += 1
        else:
            X[1] += 1
        for i in range(1, len(trace[0])):
            iat = trace[0][i] - trace[0][i-1]
            if iat > threshold:
                X.extend([0, 0])
            if trace[2][i] > 0:
                X[len(X)-2] += 1
            else:
                X[len(X)-1] += 1

    if style == "time-bursts":
        X = [0, 0]
        s = 0
        for i in range(len(trace[0])):
            timestamp = trace[0][i]
            if timestamp // interval > s:
                s = timestamp // interval
                X.extend([0, 0])
            direction = trace[2][i]
            if direction > 0:
                X[len(X)-2] += 1
            else:
                X[len(X)-1] += 1

    if X is None:
        X = sample[1].tolist()

    # pad and slice
    #padding = slice_size - (len(X) % slice_size)
    #X.extend([0]*padding)
    slices = [X[x-slice_size:x] for x in range(slice_size,len(X))]
    return slices

def generate_sample(args, samples, classes, slice_size=10, counts=1, **kwargs):
    tab_idxs = np.array(classes)[np.random.randint(0, len(classes), size=args.tc)]
    tabs = []
    for tab_idx in tab_idxs:
        while True:
            sample = samples[tab_idx][np.random.randint(0, len(samples[tab_idx]))]
            if sample[0][-1]-1 > args.lp: break
        tabs.append(sample)
    Xs, ys = [],[]
    for c in range(counts):
        sim_sample, split_points = generate_multitab_time(tabs, np.random.uniform(args.lp, min(args.up, tabs[0][0][-1]), size=len(tabs)-1))
        slices = process(sim_sample, slice_size, kwargs)
        start_idx = split_points[0][0] #// slice_size
        end_idx = np.argwhere(sim_sample[0] == tabs[0][0][-1])[0][0] #// slice_size
        #y = [0 for i in range(start_idx)]
        #y.extend([1 for i in range(start_idx, len(slices))])
        y = np.zeros((len(slices),1))
        #y[:start_idx,0] = 0
        y[start_idx:end_idx] = 1
        y[end_idx:] = 0
        Xs.append(np.array(slices)[...,np.newaxis])
        ys.append(y)
    return Xs, ys

def build_model(input_size=20):
    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(input_size,1)))
    #model.add(Bidirectional(LSTM(256, return_sequences=True)))
    #model.add(Bidirectional(LSTM(256, return_sequences=True)))
    #model.add(Reshape((5,10,1)))
    #model.add(TimeDistributed(Conv1D(64, 3)))
    #model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(input_size,1)))
    #model.add(Bidirectional(LSTM(20, dropout=0.5, activation='relu')))
    #model.add(Flatten())
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    #model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-ld", required=True)
    parser.add_argument("-tc", required=False, default=2, type=int)
    parser.add_argument("-lp", required=False, default=5, type=float)
    parser.add_argument("-up", required=False, default=10, type=float)
    args = parser.parse_args()

    slice_size = 50
    model = build_model(slice_size)
    model.summary()


    print("+ loading data")
    samples = load_data(args.ld)
    classes = list(samples.keys())
    sample_leave_out = 0

    counts = 10
    get_sample = lambda: generate_sample(args, samples, classes, slice_size, counts=counts, style="time-bursts", interval=0.05)
    
    print("+ Training BiLSTM")
    losses = []
    accuracies = []
    for epoch in range(1000):
        Xs,ys = get_sample()
        for X,y in zip(Xs,ys):
            history = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
            losses.append(history.history['loss'][0])
            accuracies.append(history.history['accuracy'][0])
            try:
                preds = model.predict(X, verbose=0)
                yhat = preds > 0.5
                print(preds.shape, yhat.shape)
                y_p = np.argwhere(y == 1)[0][0]
                yhat_p = np.argwhere(yhat == 1)[0][0]
                print(f"\tCloseness: {abs(y_p-yhat_p)} [{y_p},{yhat_p}]")
            except:
                pass
        print(f"{epoch}: avgloss {np.mean(losses[-10:])}, avgacc {np.mean(accuracies[-10:])}")
    
    print("+ Testing a few samples...")
    for i in range(10):
        X,y = get_sample()
        preds = model.predict(X, verbose=0)
        yhat = preds > 0.5
        y_p = np.argwhere(y == 1)[0][0]
        yhat_p = np.argwhere(yhat == 1)[0]
        print(f"\tCloseness: {abs(y_p-yhat_p)} [{y_p},{yhat_p}]")
