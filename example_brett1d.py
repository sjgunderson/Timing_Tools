import numpy as np
import matplotlib.pyplot
import scipy as sc
import bretthorst1d as brett

# ----- Simulated Light Curve -----
def simulate_light_curve(t, f, amps, noise_std=0.2):
    signal = np.zeros_like(t)
    signal = amps[0] * np.cos(2 * np.pi * f * t) + amps[1] * np.sin(2 * np.pi * f * t)
    noise = np.random.normal(0, noise_std, size=len(t))
    return signal + noise

np.random.seed(42)

# Time vector
N = 500
t = np.sort(np.random.uniform(0, 10, N))

# True parameters
true_freqs = 5
true_amps = [1,0]

# Events
y = simulate_light_curve(t, true_freqs, true_amps)

# Frequency Grid
freq = np.linspace(1,10,1000)

# Apply the Bretthorst algorithm
(Bamps, power, like, loglike, sigma, phat) = brett.brett1d(t, y, freq)

# Plot it in frequency units
brett.plotfreq(freq, power, sigma)

# Get the best values
(best_freq, best_freq_err) = brett.bestfreq(freq, power, sigma, t[0], t[-1])
print("Bretthorst found f = ", best_freq, " +/- ", best_freq_err)

#end of file

