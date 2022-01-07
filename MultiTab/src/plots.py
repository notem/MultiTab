import numpy as np
import matplotlib.pyplot as plt


accs = np.load('accs.npy')
dists = np.load('timedist.npy')
acc_v_th = np.load('acc_v_th.npy')
pkt_ths = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]


# pkt_th vs acc
plt.figure(figsize=(6,3))
plt.plot(pkt_ths, acc_v_th, 'b-')
plt.axis([1, 200, 0, 0.5])
plt.savefig('acc_v_th.png')
plt.clf()

# acc v offset
print(accs)
plt.figure(figsize=(6,3))
plt.plot([i+2 for i in range(len(accs))], accs, 'b-')
plt.axis([2, len(accs)+2, 0, 0.5])
plt.savefig('acc_v_offset.png')
plt.clf()

# avg_time v offset
dists = np.mean(dists,axis=1)
plt.figure(figsize=(6,3))
plt.plot([i+2 for i in range(len(dists))], dists, 'b-')
#plt.axis([2, len(dists), ])
plt.savefig('timediff_v_offset.png')
plt.clf()
