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
def box_filter(f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ...

# define the kernel of the box filter
kernel = 1 / 4 * np.array([1, 2, 1])

# filter f_noisy using the box filter


# plot filtered f and its Fourier transform
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("Box Filter")
plt.show()
# --------------------------------------------------------


# --------------------------------------------------------
# Insert solutions for 2d here
def ft_denoise(F: np.ndarray, threshold: int) -> np.ndarray:
    ...

# fourier transform f_noisy, apply fourier filter and apply inverse fourier transformation


# plot the fourier filtered f and its Fourier transform
fig, axs = plt.subplots(2, constrained_layout=True)

fig.suptitle("FT Filter")
plt.show()
# --------------------------------------------------------
