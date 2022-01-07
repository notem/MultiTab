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
        X = sample[0]*sample[1]

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

def generate_sample(args, samples, classes, seq_len=None, counts=1, **kwargs):
    """
    """
    tab_idxs = np.array(classes)[np.random.randint(0, len(classes), size=args.tc)]
    tabs = []

    for tab_idx in tab_idxs:
        while True:
            sample = samples[tab_idx][np.random.randint(0, len(samples[tab_idx]))]
            if sample[0][-1]-1 > args.lp: break
        tabs.append(sample)

    res = []
    for c in range(counts):
        r = np.random.uniform(args.lp, min(args.up, tabs[0][0][-1]), size=len(tabs)-1)
        sim_sample, split_points = generate_multitab_time(tabs,r) 

        X = process(sim_sample, **kwargs)
        start_pkt_idx = split_points[0][0]
        slices, y = do_slices(X, kwargs["slice_size"], kwargs["stride_size"], start_pkt_idx)

        if seq_len is not None:
            if len(slices) > seq_len:
                slices = slices[:seq_len]
            elif len(slices) < seq_len:
                padlen = seq_len - len(slices)
                if isinstance(slices[0], list):
                    slices += [[0]*len(slices[0])]*padlen
                else:
                    slices += [0]*padlen

        res.append((slices, y, start_pkt_idx))

    return res


class CNNBLSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_units=500, 
                     f_nums=[32, 64], k_sizes=[5, 5], 
                     layers=3, dropout_perc=0.5):
        super(CNNBLSTM, self).__init__()
        self.n_features = n_features
        self.n_hidden = hidden_units # number of hidden states
        self.n_layers = layers # number of LSTM layers (stacked)
        self.dropout_perc = dropout_perc
        self.conv1_1 = torch.nn.Conv1d(n_features, f_nums[0], k_sizes[0], padding=((k_sizes[0]-1)//2))
        #self.conv1_2 = torch.nn.Conv1d(f_nums[0], f_nums[1], k_sizes[1], padding=((k_sizes[1]-1)//2))
    
        self.l_lstm = torch.nn.LSTM(input_size = f_nums[0], 
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
        #x = torch.relu(self.conv1_2(x))
        #print(x.size())
        x = x.permute(0,2,1)
        #print(x.size())
        lstm_out, h = self.l_lstm(x, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = h
        predictions = self.l_linear(lstm_out).squeeze()
        predictions = torch.sigmoid(predictions)
        return predictions




class BLSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_units=70, layers=2, dropout_perc=0.5):
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

def naive_predict(slices, sp):
    counts = np.sum(slices, axis=1)
    mi, ma = np.amin(counts), np.amax(counts)
    best_th = -1
    best_dst = 10000000000
    for th in np.linspace(mi,ma,100):
        guess = len(counts)
        for i in range(len(counts)):
            if counts[i] > th:
                guess = i
                break
        dst = abs(sp - guess)
        if dst < best_dst:
            best_dst = dst
            best_th = th
    return best_dst, best_th
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-ld", required=True)
    parser.add_argument("-tc", required=False, default=2, type=int)
    parser.add_argument("-lp", required=False, default=2, type=float)
    parser.add_argument("-up", required=False, default=20, type=float)
    args = parser.parse_args()

    n_features = 2
    process_kwargs = {"style": "time-bursts", "interval": 0.25, 
                      "slice_size": n_features, "stride_size": n_features}
    #process_kwargs = {"style": "tiktok"}
    #process_kwargs = {"style": None}

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)  

    # create NN
    mv_net = BLSTM(n_features).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=0.001)
    
    train_episodes = 10000
    threshold = 0.5

    print("+ loading data")
    samples = load_data(args.ld)
    classes = list(samples.keys())

    counts = 1
    chunk_size = 100
    get_sample = lambda: generate_sample(args, samples, classes, counts=counts, **process_kwargs)
    
    print("+ Training BiLSTM")
    losses = []
    accuracies = []
    thresholds = []
    dists = []
    ndists = []
    nthresholds = []
    for epoch in range(train_episodes):
        res = []
        for _ in range(chunk_size):
            res.extend(get_sample())
        c = 0
        for X,y,sp in res:
            #print(f'[step {epoch}] prog : {c}/{len(res)}', end='\r')
            c += 1

            x_batch = torch.tensor([X], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y, dtype=torch.float32).squeeze().to(device)

            mv_net.init_hidden(1, device=device)
            output = mv_net(x_batch)
  
            try:
                loss = criterion(output.view(-1), y_batch)  
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(mv_net.parameters(), 1.0)
                optimizer.step()        
                optimizer.zero_grad() 
            except Exception as e:
                print(f'[step {epoch}] encountered error when calculating loss, {Xs[0].shape} {ys[0].shape}')
                print(e)
                continue

            out = output.cpu().detach().numpy().flatten()
            #print(y.flatten(), out)
            best_dist = 10000000
            best_th = 0.5
            try:
                idx_true = np.argwhere(y == 1)[0]
            except:
                idx_true = -1
            for th in np.linspace(0.5,1,20):
                preds = out > th
                try:
                    idx_pred = np.argwhere(preds == 1)[0]
                except:
                    idx_pred = -1
                dist = np.abs(idx_pred-idx_true)
                if dist < best_dist:
                    best_dist = dist
                    best_th = th
            
            dists.append(best_dist)
            thresholds.append(best_th)

            ndist, nth = naive_predict(X, idx_true)
            ndists.append(ndist)
            nthresholds.append(nth)

            losses.append(loss.item())
        print('-----------------------')
        print(f'[step {epoch}] loss : {np.mean(losses[-1000:])},')
        print(f'dist: {np.mean(dists[-1000:])}, th: {np.mean(thresholds[-1000:])}, ndist: {np.mean(ndists[-1000:])}  th: {np.mean(nthresholds[-1000:])}')
        torch.save(mv_net.state_dict(), "model.pt")

    torch.save(mv_net.state_dict(), "model.pt")
    np.save('losses.npy', losses)
    
    #print("+ Testing a few samples...")
    #for i in range(10):
    #    X,y = get_sample()
    #    preds = model.predict(X, verbose=0)
    #    yhat = preds > 0.5
    #    y_p = np.argwhere(y == 1)[0][0]
    #    yhat_p = np.argwhere(yhat == 1)[0]
    #    print(f"\tCloseness: {abs(y_p-yhat_p)} [{y_p},{yhat_p}]")
