"""Insert your solutions for exercise 2 here."""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# sampling details of the function f
sf = 1000  # sampling frequency
T = 1 / sf  # sampling period
N = 1200  # number of samples
x = np.arange(N) * T  # time vector

# vector containing the frequencies for plotting in fourier space
# we can shift the frequencies using fftshift so that 0 is in the middle of the plot
# [-500., -499.16666667, ..., 0, ...,  498.33333333,  499.16666667]
freq = np.fft.fftshift(np.fft.fftfreq(N, d=T))

# create a function f as a superposition of two cosine waves
f = 0.5 * np.cos(2 * np.pi * 30 * x) + 0.8 * np.cos(2 * np.pi * 80 * x)

# create noisy version of f for exercise 2c
noise = np.random.randn(N)
f_noisy = f + noise

# plot both functions to demonstrate the plotting
fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
axs[0].set_title("f")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].plot(x, f, "r")
axs[1].set_title("f with noise")
axs[1].set_xlabel("x")
axs[1].set_ylabel("$f_{noisy}(x)$")
axs[1].plot(x, f_noisy, "b")
fig.suptitle("Plotting Example")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2a and 2b here
def fourier_transform(f: np.ndarray, inv: bool = False) -> np.ndarray:
    #converting f into a array
    f = np.asarray(f, dtype=np.complex128).ravel()
    N = f.shape[0]
    if N == 0:
        return np.array([])
    
    # x runs the all of the indexes of the input signal
    # u is the column version 
    x = np.arange(N)
    u = x.reshape((N, 1))

    # chooses sign for forward or inverse DTF
    sign = -1 if not inv else 1
    # creating matrix of u*x products (= N*N matrix)
    exponent = sign * 2j * np.pi * (u @ x.reshape(1, -1)) / N
    # complex exponential matrix
    W = np.exp(exponent)

    if not inv:
        # forward DTF: multiplying the N*N matrix by the vector 
        return W.dot(f)
    else:
        # inverse DTF: scaling by 1/N as given in the formula
        return (1.0 / N) * W.dot(f)

# use the function to compute the DTF and then to get the fourier transform F of the function f
F = fourier_transform(f, inv=False)
# rearanging the array so the negative and positive frquencies are in the correct order to be plotted
F_shifted = np.fft.fftshift(F)
# F was normalized, to reflect the actual amplitudes of the original cosines. The frequencies remained the same, only the yaxis scale changed.
# DTF: if we look at a pure cosine the magnitude of each frequency component is proportional to N/2 --> dividing F_shifted by N/2 to rescale the magnitude axis
F_shifted_norm = F_shifted / (N/2)



# plot f and F
fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
fig.suptitle("Fourier Transform")

# Time-domain plot f
axs[0].plot(x, f, "r")
axs[0].set_title("Signal f(x)")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# Frequency-domain plot F
axs[1].plot(freq, np.abs(F_shifted_norm), "b")
axs[1].set_title("Magnitude spectrum |F(f)|")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("|F(f)|")
axs[1].grid(True)

plt.show()

# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2c here

# task: method to smooth a noisy 1D signal in the time domain
# (moving average (box) filter using convolution to reduce the noise)
def box_filter(f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # we have to make sure that the input signal (f) and kernel are numpy arrays
    f = np.asarray(f)
    kernel = np.asarray(kernel)
    
    k = len(kernel) # here the length of the filter kernel would be 3
    # we need padding because without it the convolution would go out of bounds or shorten signal
    # for an odd-length kernel: number of neighbors on each side (k-1)/2 == k//2 -> symmetric padding
    pad = k//2
    
    # task: pad with zero at both sides using np.pad to avoid issues at the edges of the signal
    # this has the effect of output length == input length and that the filter is centered at each position
    f_padded = np.pad(f, pad, mode="constant")
    
    # initializing output array for filtered signal
    filtered = np.zeros_like(f,dtype=float)
    
    # task: the filter should be applied using convolution: implemented ourselves
    # for each signal position i: multiply the kernel with the corresponding signal segment and sum the result
    for i in range(len(f)):
        filtered[i]=np.sum(f_padded[i:i+k]*kernel)
    return filtered

# define the kernel of the box filter
# weighted moving average: central value: higher weight than neighboring samples
# task: use provided kernel
kernel = 1 / 4 * np.array([1, 2, 1])

# filter f_noisy using the box filter
f_filtered = box_filter(f_noisy, kernel)

# Fourier transform of filtered signal
# analyze effect of the filter in frequency domain
F_filtered = fourier_transform(f_filtered, inv=False)
# shifting zero frequency to center for better visualisation
F_filtered_shifted = np.fft.fftshift(F_filtered)

# plot filtered f and its Fourier transform
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("Box Filter")

# Time-domain plot
axs[0].plot(x, f_filtered)
axs[0].set_title("Signal f(x)")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# Frequency-domain plot (filtered signal)
# plotting magnitude spectrum of filtered signal
axs[1].plot(freq, np.abs(F_filtered_shifted), "b")
axs[1].set_title("Magnitude spectrum |F(f)|")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("|F(f)|")
axs[1].grid(True)


plt.show()

# How would you rate the filtered result? 
  # How does the plot look?
  # The time-domain plot (Signal f(x)) shows the smoothed signal when applying the box_filter
  # The signal oscillates between roughly -2 and 2 because of the original cosine waves plus some remaining noise...
  # It is however much smoother compared to the noisy signal, the signal becomes easier to see, but main peaks are still visible
  
  # The frequency-domain plot shows the fourier magnitude of the filtered signal with four main peaks symmetric around 0 Hz (correspond to two cosine waves in original signal...)
  # The rest of spectrum is relativelly small (=remaining low-frequency noise) --> small values == removed noise

# rating:
# The box_filter function smooths the noisy signal well and effectively since it is reducing the high frquency noise.
# But this can lead to reducing higher frequencies of the original signal so that smaller/fine details get lost in the process.
# All in all the filter is simple but effective to reduce noise but it is not absolutely ideal when we want to keep all details in the signal...


# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2d here

# remove noise in the Fourier domain by zeroing small coefficients
# F is the Fourier transform of the noisy signal
# threshold is the magnitude below which frequencies are considered noise
def ft_denoise(F: np.ndarray, threshold: int) -> np.ndarray:
    # copying to avoid changes to the original Fourier coefficients
    F_filtered = F.copy()
    # sets all coefficients under threshold to 0 (removes small-magnitude Fourier coefficients, reduces noise)
    F_filtered[np.abs(F_filtered)<threshold]=0
    # inverse fourier transformation to fet denoised signal in time domain
    f_reconstructed = fourier_transform(F_filtered, inv = True)
    # reconstructed signal after inverse fourier transformation
    return f_reconstructed

# fourier transform f_noisy, apply fourier filter and apply inverse fourier transformation
# fourier transform f_noisy
F_noisy = fourier_transform(f_noisy)

# Denoise in Fourier domain
# define threshold and give it as argument
threshold = 0.1 * np.max(np.abs(F_noisy)) # 0.1 * max(|F_noisy|) very simple choice of noise filter
f_denoised = ft_denoise(F_noisy, threshold) #reconstructing applying fourier filter and inverse fourier transformation ...

# Fourier transform of denoised signal
F_denoised = fourier_transform(f_denoised)
F_denoised_shifted = np.fft.fftshift(F_denoised) #shifting zero frequency to center (as before)

# plot the fourier filtered f and its Fourier transform
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("FT Filter")

# Time-domain plot
axs[0].plot(x, f_denoised)
axs[0].set_title("Signal f(x) (reconstructed)")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# Frequency-domain plot
axs[1].plot(freq, np.abs(F_denoised_shifted))
axs[1].set_title("Magnitude spectrum |F(f)|")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("|F(f)|")
axs[1].grid(True)

plt.show()


# task: Explain your method
# The Noise is reduced in the Fourier domain by setting all coefficients that are smaller than the defined, fixed threshold to 0.
# This has as consequence that the most important frequencies (so to say the original cosine waves) stay visible, while most of the samller noise is removed.
# By applying the inverse Fourier Transformation the smoothed signal is reconstructed back in the time domain. 


# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2e here
# task: Which method was better at reducing the noise? 
# Explain both your answer and the reason why the method performed better. 
# Also explain when this method might not be applicable.

# Our Answer:
# The Fourier-based denoising method (2d) performed better than the box_filter (2c) in terms of noise reduction.

# This is because the box_filter smooths the signal in the time domain by averaging the neighboring values. And while this in fact does reduce the noise it also smooths over the original signal 
# and can therefore lead to reducing higher frequencies of the original, the true signal.
# The 2c option leads to signal details being lost.

# The Fourier-based, 2d option, however works in the frequency domain and directly removes the frequency compoinents with too small magnitudes, which we assume to be mainly noise.
# Therefore the important frequencies of the original signal are preserved while noise is removed...

# The problem with the Fourier-based method could be that the noise overlaps too much with the original signal frequencies or that they aren´t a few (in this case: 4) dominant frequencies in the signal.
# Then using the 2d method would remove parts of the actual signal or maybe keep too much noise. That´s why in such cases the method would not be applicable.

# --------------------------------------------------------