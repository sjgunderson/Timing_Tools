import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

"""
    bretthorst1d(x, y, f)

1-dimensional periodogram of unevenly sampled data using Bretthorst's Bayesian algorithm.
This is a Python implementation of the Julia alogrithm written by Paul Barrett:
https://github.com/barrettp/Periodogram

Required variables:
x = time coordinates
y = events
f = frequency grid

Optional parameters:
zloge = center point for likelihood

Outputs:
B = [cosine amplitude, sine amplitude]
h2bar = power^2
st = likelihood
stloge = log likelihood
sigma = standard deviation
phat = ...need to check with Paul


Notes:
Event values MUST have their median taken out:
	y = y - np.median(y)
Currently this is done automatically, but there 
shouldn't be a problem the user enters an array
that already follows this...hopefully


"""
def brett1d(x, y, freq, zloge=0.0):
  M, N = len(x), len(freq)
  B, h2bar, st = np.zeros((N,2)), np.zeros(N), np.zeros(N)
  stloge, sig, phat = np.zeros((N,2)), np.zeros(N), np.zeros(N)

  #First let's remove the median so we don't have to when using
  #the algorithm
  y = y - np.median(y)
  y = np.atleast_2d(y).T #Need y to be a column vector for matrix math
  
  for j in range(N):
    """
    Important notes on object shapes:
    y = (2,)
    Gj = (N,2)
    GjTGj = (2,2)
    lj = (2,1) <-- Eigen values
    ej = (2,2) <-- Eigen vectors
    hj = (2,1)
    B[j,:] = (2,1)
    h2bar[j] = (1,)
    
    """

    #Now we start the algorithm!
    holdcos = np.cos(2*np.pi*freq[j]*x)
    holdsin = np.sin(2*np.pi*freq[j]*x)
    Gj = np.column_stack((holdcos, holdsin))

    """
    For this algorithm, we need to do some extra numpy steps
    because there are no built in functions for Julia's
    Hermitian function.
    """
    #First we take the matrix product
    GjTGj = Gj.T @ Gj
    #Now we need to copy over only the upper triangle
    upper_triangle = np.triu(GjTGj)
    # Fill the lower triangle with the conjugate transpose of the upper triangle
    # This creates a Hermitian matrix based on the upper triangle
    hermitian_GjTGj = upper_triangle + np.conjugate(upper_triangle.T) - np.diag(np.diag(upper_triangle))
    
    #Now we can continue with the algorithm
    lj, ej = np.linalg.eig(hermitian_GjTGj)
    hj = (ej @ Gj.T @ y)/np.sqrt(lj[:,None])
    
    B[j,:] = np.reshape((ej @ hj)/np.sqrt(lj[:,None]),2)
    h2bar[j] = ((hj.T @ hj)/2).item()
    stl = np.log(1 - (hj.T @ hj)/(y.T @ y)) * ((2 - M)/2)
    stloge[j] = stl
    if abs(zloge) != 0:
      st[j] = np.exp(stl - zloge)
    else:
      st[j] = 0.0
      
    sig[j] = np.sqrt(((y.T @ y).item() - (hj.T @ hj).item())/(M-4))
    phat[j] = (hj.T @ hj).item() * st[j]
    
    return (B, h2bar, st, stloge, sig, phat)

"""
    plotfreq(f, power, sigma)
    
Basic Plotting function for displaying the
Bretthorst periodogram

Required inputs:
f = frequency grid used for brett1d
power = power^2 output from brett1d
sigma = sigma output from brett1d

Optional Parameters
smooth = width value for Gaussian smoothing (default = 0)
period = Switch for plotting periods (default = 0)

"""
def plotfreq(f, power, sigma, smooth=0, period=0):
  fig = plt.figure(figsize=(15,5))
  if smooth > 0:
    SNR = sc.ndimage.gaussian_filter1d(np.sqrt(power)/sigma, smooth)
  else:
    SNR = np.sqrt(power)/sigma

  if period > 0:
    x = 1/f
    xlabel = "Period"
  else:
    x = f
    xlabel = "Frequency"
    
  plt.plot(x, SNR, marker="",ls="-")
  plt.ylabel("Bretthorst S/N",fontsize=18)
  plt.xlabel(xlabel,fontsize=18)

  plt.minorticks_on()
  plt.tick_params(which="both", direction="in", right=True, top=True, labelsize=16)
  
  
  plt.show()

"""
    plotfreq(f, power, sigma)
    
Basic function for computing best period from
the Bretthorst periodogram

Required inputs:
f = frequency grid used for brett1d
power = power^2 output from brett1d
sigma = sigma output from brett1d
tstart = start of the lightcurve
tend = end of the lightcurve

Optional Parameters
smooth = width value for Gaussian smoothing (default = 0)
period = Switch to output the period

Outputs = (frequency or period, uncertainty)

Warning: If the frequency grid is not samplied sufficiently, smoothing can distort
the period and uncertainties!!!
"""
def bestfreq(f, power, sigma, tstart, tend, smooth=0, period=0):
  if smooth > 0:
    print("Warning: If the frequency grid is not samplied sufficiently, smoothing can distort ")
    print("the period and uncertainties!")
    SNR = sc.ndimage.gaussian_filter1d(np.sqrt(power)/sigma, smooth)
  else:
    SNR = np.sqrt(power)/sigma
    
  index = np.argmax(SNR)
  DeltaT = (tstart - tend)
  
  freq = f[index]
  sigmaf = 1.1/(np.sqrt(2)* SNR[index] * DeltaT)
  
  if period > 0:
    print("Returning Period")
    return (1/freq, sigmaf / freq**2)
  else:
    print("Returning Frequency")
    return (freq, sigmaf)



#end of file
