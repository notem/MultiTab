import os
from matplotlib import pyplot as plt
import numpy as np


root_dir = "/home/shared-data/datasets/AWF100/awf_cw/"
trace_1 = os.path.join(root_dir, "netflix.com/5")
trace_2 = os.path.join(root_dir, "twitter.com/1")
perc_overlap = 0.5


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


with open(trace_1, 'r') as fi:
    t1 = load_trace(fi)
with open(trace_2, 'r') as fi:
    t2 = load_trace(fi)


def generate_multitab(traces, perc_overlap):
    sim_trace = [traces[0][0].copy(), traces[0][1].copy(), traces[0][2].copy()]
    time_offset = 0
    for i in range(len(traces)-1):
        merge_point = int(len(traces[0][0])*perc_overlap)
        time_offset += traces[i][0][merge_point]
        sim_trace[0].extend([timestamp+time_offset for timestamp in traces[i+1][0]])
        sim_trace[1].extend(traces[i+1][1])
        sim_trace[2].extend(traces[i+1][2])
    indxs = np.argsort(sim_trace[0])
    sim_trace = np.array([np.array(sim_trace[0])[indxs].tolist(),
                          np.array(sim_trace[1])[indxs].tolist(), 
                          np.array(sim_trace[2])[indxs].tolist()])
    return sim_trace

sim_trace = generate_multitab([t1, t2], perc_overlap)

def cumul_plot(t1, t2, t3, perc_overlap):
    x1 = t1[0]
    y1 = [-t1[1][0]*t1[2][0]]
    for i in range(1, len(t1[0])):
        y1.append(y1[-1] - (t1[1][i]*t1[2][i]))
    x2 = t2[0]
    y2 = [-t2[1][0]*t2[2][0]]
    for i in range(1, len(t2[0])):
        y2.append(y2[-1] - (t2[1][i]*t2[2][i]))
    x3 = t3[0]
    y3 = [-t3[1][0]*t3[2][0]]
    for i in range(1, len(t3[0])):
        y3.append(y3[-1] - (t3[1][i]*t3[2][i]))

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Cumulative Sums v. Time')
    ax1.plot(x1, y1)
    ax1.set_title('netflix.com')
    ax2.plot(x2, y2)
    ax2.set_title('twitter.com')
    ax3.plot(x3, y3)
    true_split = t1[0][int(perc_overlap*len(t1[0]))]
    ax3.axvline(x=true_split, color='red', linestyle='dashed', label='Join point')
    ax3.set_title('Simulated Join @ t={:.2f}'.format(true_split))
    plt.tight_layout()
    plt.savefig('sim2.png')

cumul_plot(t1, t2, sim_trace, perc_overlap)
