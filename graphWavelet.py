"""
Instantiate and plot wavelet filter banks on graphs

Author: Shashwat Shukla
Date: 2nd June 2020
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils

# Meyer wavelets on a ring
G = graphs.Ring(400)
G.estimate_lmax()
g = filters.Meyer(G, Nf=6)
fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of Meyer wavelets')

DELTA = 255
s = g.localize(DELTA)
fig = plt.figure(figsize=(10, 2.5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    _ = ax.set_title('Wavelet {}'.format(i + 1))
    ax.set_axis_off()
fig.tight_layout()
plt.show()


# Mexican hat wavelets on a torus
G = graphs.Torus()
G.estimate_lmax()
g = filters.MexicanHat(G, Nf=6)
fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of Mexican Hat wavelets')

DELTA = 255
s = g.localize(DELTA)
fig = plt.figure(figsize=(10, 2.5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    _ = ax.set_title('Wavelet {}'.format(i + 1))
    ax.set_axis_off()
fig.tight_layout()
plt.show()
