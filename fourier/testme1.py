'''
Created on Jun. 12, 2021

@author: zollen
@url: https://www.youtube.com/watch?v=O0Y8FChBaFU
'''

from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

warnings.filterwarnings('ignore')

sb.set_style('whitegrid')

Fs = 2000        # sampling frequency
f0 = 100         # signal frequency
N = int(Fs / f0) # number of samples

tstep = 1 / Fs   # sample time interval
t = np.linspace(0, (N-1) * tstep, N)  # time step

fstep = Fs / N   # frequency interval
f = np.linspace(0, (N-1) * fstep, N)  # frequency step

y = 1 * np.sin(2 * np.pi * f0 * t)

X = fft(y)
print(X)
X_mag = np.abs(X) / N
print(X_mag)

f_plot = f[0:int(N/2) + 1]
X_mag_plot = 2 * X_mag[0:int(N/2) + 1]
X_mag_plot[0] = X_mag_plot[0] / 2    # Note: DC component does not need to multiply by 2


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.set_xlabel('time (s)')
ax1.set_xlim(0, t[-1])
ax1.plot(t, y, marker='.')
ax2.set_xlabel('freq (Hz)')
ax2.set_xlim(0, f_plot[-1])
ax2.plot(f_plot, X_mag_plot, marker='.')
plt.tight_layout()
plt.show()