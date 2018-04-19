# boltzman-distribution-
import csv

list1 = []
list2 = []

with open('y.txt', 'r') as file_x:
    content = file_x.readlines()
    for x in content: 
        row = x.split()
        list1.append(row[0])
    
with open('x.txt', 'r') as file_y:
    content = file_y.readlines()
    for x in content: 
        row = x.split()
        list2.append(row[0])
        
import numpy as np
y_array = np.asarray(list1)
x_array = np.asarray(list2)



from scipy import fftpack
from matplotlib import pyplot as plt
# The FFT of the signal
sig_fft = fftpack.fft(y_array)
period = 5

plt.plot(x_array, y_array, label='Original signal')


# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(y_array.size, d=0.02)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [DAY]')
plt.ylabel('Flux')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection


high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(x_array, y_array, label='Original signal')
plt.plot(x_array, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')
plt.show()
