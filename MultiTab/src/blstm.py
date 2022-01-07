import torch
from dataprep import load_data, generate_multitab_time
from argparse import ArgumentParser
import numpy as np
from torchsummary import summary


def process(sample, slice_size=2, stride_size=2, style='directional-bursts', **kwargs):
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
        for i in range(len(sample[0])):
            timestamp = sample[0][i]
            if timestamp // kwargs["interval"] > s:
                s = timestamp // kwargs["interval"]
                X.extend([0, 0])
            direction = sample[2][i]
            if direction > 0:
                X[len(X)-2] += 1
            else:
                X[len(X)-1] += 1

    if style == "direction":
        X = sample[1]

    if style == "tiktok":
        X = sample[0]*sample[2]

    if X is None:
        X = sample.transpose((1,0))
        return X

    return X

def do_slices(X, slice_size, stride_size, split_point=-1):
    # pad and slice
    slices = []
    y = []
    tot_pkts = 0
    for i in range(slice_size, len(X), stride_size):
        b = X[i-slice_size:i]
        slices.append(b)
        tot_pkts += np.sum(np.abs(b))
        if split_point <= tot_pkts:
            y.append(1)
        else:
            y.append(0)
    slices = np.array(slices)
    if len(slices.shape) < 2:
        slices = slices[...,np.newaxis]
    return slices, np.array(y)

def generate_multitab_time(traces, offsets):
    get_time_overlap = lambda i: offsets[i] if hasattr(type(offsets), '__iter__') else offsets
    sim_trace = [traces[0][0].copy(), traces[0][1].copy(), traces[0][2].copy()]
    time_offset = 0
    time_merges = []
    split_points = []
    for i in range(len(traces)-1):
        z = len(sim_trace[0])
        time_offset = get_time_overlap(i)
        tmerge_point = -1
        for j in range(len(traces[i][0])):
            if traces[i][0][j] > time_offset:
                tmerge_point = j
                break
        sim_trace[0].extend([timestamp+time_offset for timestamp in traces[i+1][0]])
        sim_trace[1].extend(traces[i+1][1])
        sim_trace[2].extend(traces[i+1][2])
        indxs = np.argsort(sim_trace[0])
        time = time_offset+traces[i+1][0][0]
        time_merges.append(time)
    sim_trace = np.array([np.array(sim_trace[0])[indxs].tolist(),
                          np.array(sim_trace[1])[indxs].tolist(), 
                          np.array(sim_trace[2])[indxs].tolist()])
    time = np.amin(time_merges)
    merge_point = np.argwhere(sim_trace[0] == time)[0]
    return sim_trace, merge_point

def generate_sample(args, up, lp, samples, classes, seq_len=None, counts=1, **kwargs):
    """
    """
    #tab_idxs = np.array(classes)[np.random.randint(0, len(classes), size=args.tc)]
    first_sample = None
    t = np.random.randint(1,5)
    for i in range(10000):
        c = np.array(classes)[np.random.randint(0, len(classes))]
        first_sample = samples[c][np.random.randint(0, len(samples[c]))]
        if first_sample[0][-1]-1 > lp: break
    tabs = []
    tabs.append(first_sample)
    while len(tabs) < t+1:
        c = np.array(classes)[np.random.randint(0, len(classes))]
        sample = samples[c][np.random.randint(0, len(samples[c]))]
        tabs.append(sample)
    res = []
    for c in range(counts):
        r = np.random.uniform(lp, min(up, tabs[0][0][-1]), size=len(tabs)-1)
        sim_sample, split_points = generate_multitab_time(tabs,r) 

        X = process(sim_sample, **kwargs)
        start_pkt_idx = split_points[0]
        slices, y = do_slices(X, kwargs["slice_size"], kwargs["stride_size"], start_pkt_idx)
        y = np.zeros(len(slices))
        y[start_pkt_idx:] = 1

        if seq_len is not None:
            if len(slices) > seq_len:
                slices = slices[:seq_len]
            elif len(slices) < seq_len:
                padlen = seq_len - len(slices)
                if isinstance(slices[0], list):
                    slices += [[0]*len(slices[0])]*padlen
                else:
                    slices += [0]*padlen

        res.append((slices, y, start_pkt_idx, sim_sample))

    return res


class CNNBLSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_units=100, 
                     f_nums=[16, 32], k_sizes=[9, 9], 
                     layers=2, dropout_perc=0.5):
        super(CNNBLSTM, self).__init__()
        self.n_features = n_features
        self.n_hidden = hidden_units # number of hidden states
        self.n_layers = layers # number of LSTM layers (stacked)
        self.dropout_perc = dropout_perc
        self.conv1_1 = torch.nn.Conv1d(n_features, f_nums[0], k_sizes[0], padding=((k_sizes[0]-1)//2))
        self.conv1_2 = torch.nn.Conv1d(f_nums[0], f_nums[0], k_sizes[0], padding=((k_sizes[0]-1)//2))
        self.pool1 = torch.nn.MaxPool1d(k_sizes[0], padding=((k_sizes[0]-1)//2), stride=1)
        self.conv_dropout = torch.nn.Dropout(0.1)
        self.conv2_1 = torch.nn.Conv1d(f_nums[0], f_nums[1], k_sizes[1], padding=((k_sizes[1]-1)//2))
        self.conv2_2 = torch.nn.Conv1d(f_nums[1], f_nums[1], k_sizes[1], padding=((k_sizes[1]-1)//2))
        self.pool2 = torch.nn.MaxPool1d(k_sizes[1], padding=((k_sizes[1]-1)//2), stride=1)
        self.conv_dropout = torch.nn.Dropout(0.1)
    
        self.l_lstm = torch.nn.LSTM(input_size = f_nums[-1], 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True, 
                                 dropout=self.dropout_perc,
                                 bidirectional = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*2, 1)
        
    
    def init_hidden(self, batch_size, device=None):
        # even with batch_first = True this remains same as docs
        self.hidden_state = torch.zeros(self.n_layers*2,batch_size,self.n_hidden)
        self.cell_state = torch.zeros(self.n_layers*2,batch_size,self.n_hidden)
        if device != None:
            self.hidden_state = self.hidden_state.to(device)
            self.cell_state = self.cell_state.to(device)
    
    
    def forward(self, x):
        x = x.permute(0,2,1)
        #print(x.size())
        x = torch.relu(self.conv1_1(x))
        #print(x.size())
        x = torch.relu(self.conv1_2(x))
        #print(x.size())
        #x = self.pool1(x)
        #print(x.size())
        x = self.conv_dropout(x)
        x = torch.relu(self.conv2_1(x))
        #print(x.size())
        x = torch.relu(self.conv2_2(x))
        #print(x.size())
        #x = self.pool2(x)
        #print(x.size())
        x = self.conv_dropout(x)
        x = x.permute(0,2,1)
        #print(x.size())
        lstm_out, h = self.l_lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = h
        predictions = self.l_linear(lstm_out).squeeze()
        predictions = torch.sigmoid(predictions)
        return predictions




class BLSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_units=30, layers=2, dropout_perc=0.5):
        super(BLSTM, self).__init__()
        self.n_features = n_features
        self.n_hidden = hidden_units # number of hidden states
        self.n_layers = layers # number of LSTM layers (stacked)
        self.dropout_perc = dropout_perc
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True, 
                                 dropout=self.dropout_perc,
                                 bidirectional = True)
        self.l_linear = torch.nn.Linear(self.n_hidden*2, 1)
        
    
    def init_hidden(self, batch_size, device=None):
        # even with batch_first = True this remains same as docs
        self.hidden_state = torch.zeros(self.n_layers*2,batch_size,self.n_hidden)
        self.cell_state = torch.zeros(self.n_layers*2,batch_size,self.n_hidden)
        if device != None:
            self.hidden_state = self.hidden_state.to(device)
            self.cell_state = self.cell_state.to(device)
    
    
    def forward(self, x):        
        lstm_out, h = self.l_lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = h
        predictions = self.l_linear(lstm_out).squeeze()
        predictions = torch.sigmoid(predictions)
        return predictions

def tmp(output, X, threshold, slice_size, stride_size):
    preds = output > threshold
    guess_b = -1
    for i in range(5, len(preds)):
        tmp = True
        for j in range(5):
            if not preds[i-5+j]:
                tmp = False
                break
        if tmp:
            guess_b = i-5
            break
    guess = 0
    for i in range(len(X)):
        if i == guess:
            guess += np.sum(np.abs(X[i])) // 2
            break
        else:
            guess += np.sum(np.abs(X[i]))
    return guess


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-ld", required=True)
    parser.add_argument("-tc", required=False, default=2, type=int)
    parser.add_argument("-lp", required=False, default=2, type=float)
    parser.add_argument("-up", required=False, default=30, type=float)
    args = parser.parse_args()

    n_features = 1
    #process_kwargs = {"style": "time-bursts", "interval": 0.5, 
    #                  "slice_size": n_features, "stride_size": n_features}
    process_kwargs = {"style": "tiktok",
                      "slice_size": n_features, "stride_size": n_features}
    #process_kwargs = {"style": None}

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)  

    # create NN
    mv_net = CNNBLSTM(n_features).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adamax(mv_net.parameters(), lr=0.003)
    
    train_episodes = 10000
    threshold = 0.5

    print("+ loading data")
    samples = load_data(args.ld)
    classes = list(samples.keys())

    counts = 1
    chunk_size = 10

    lp = args.lp
    
    print("+ Training BiLSTM")
    losses = []
    accuracies = []
    thresholds = []
    dists = []
    for epoch in range(train_episodes):
        up = min(lp+1+len(losses)/300, args.up)

        get_sample = lambda: generate_sample(args, up, lp, samples, classes, counts=counts, **process_kwargs)
        res = []
        for _ in range(chunk_size):
            res.extend(get_sample())
        c = 0
        for X,y,sp,_ in res:
            #print(f'[step {epoch}] prog : {c}/{len(res)}', end='\r')
            c += 1

            x_batch = torch.tensor([X], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y, dtype=torch.float32).squeeze().to(device)

            mv_net.init_hidden(1, device=device)
            output = mv_net(x_batch)
  
            try:
                loss = criterion(output.view(-1), y_batch)  
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mv_net.parameters(), 0.5)
                optimizer.step()        
                optimizer.zero_grad() 
            except Exception as e:
                print(f'[step {epoch}] encountered error when calculating loss, {X.shape} {y.shape}')
                print(e)
                continue

            out = output.cpu().detach().numpy().flatten()
            #print(y.flatten(), out)
            #idx_pred = tmp(out, X, threshold)
            preds = np.array(out > threshold, dtype=np.int32)
            try:
                idx_pred = np.argwhere(preds == 1)[0]
            except:
                idx_pred = -1
            try:
                idx_true = np.argwhere(y == 1)[0]
            except:
                idx_true = -1
            #idx_true = sp
            #print(idx_pred, idx_true)
            dist = np.abs(idx_pred-idx_true)
            dists.append(dist)

            losses.append(loss.item())
        print('-----------------------')
        print(up, preds, y)
        print(f'[step {epoch}] loss : {np.mean(losses[-chunk_size:])}, dist: {np.mean(dists[-chunk_size:])}')
        print('-----------------------')
        torch.save(mv_net.state_dict(), "model5.pt")

    torch.save(mv_net.state_dict(), "model5.pt")
    np.save('losses5.npy', losses)
    
    #print("+ Testing a few samples...")
    #for i in range(10):
    #    X,y = get_sample()
    #    preds = model.predict(X, verbose=0)
    #    yhat = preds > 0.5
    #    y_p = np.argwhere(y == 1)[0][0]
    #    yhat_p = np.argwhere(yhat == 1)[0]
    #    print(f"\tCloseness: {abs(y_p-yhat_p)} [{y_p},{yhat_p}]")
